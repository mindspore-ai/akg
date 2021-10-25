/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

using OpAttr = Map<std::string, NodeRef>;

#define TOPI_ONE_INPUT_ONE_ATTR_CALL(ins, rv, fn, get_attr) \
  do {                                                      \
    auto inputs = ins[0].operator Array<NodeRef>();         \
    CHECK_EQ(inputs.size(), 1);                             \
    CHECK(inputs[0]->IsInstance<TensorNode>());             \
    auto attrs = ins[1].operator OpAttr();                  \
    CHECK(!attrs.empty());                                  \
    *rv = fn(Downcast<Tensor>(inputs[0]), get_attr(attrs)); \
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
  auto orig_dtype = dtype;
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

  if (orig_dtype == Float(16)) {
    a_tensor = topi::cast(a_tensor, Float(32));
    b_tensor = topi::cast(b_tensor, Float(32));
    c_tensor = topi::cast(c_tensor, Float(32));
    if (d_tensor.defined()) {
      d_tensor = topi::cast(d_tensor, Float(32));
    }
    dtype = Float(32);
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

  if (orig_dtype == Float(16)) {
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

TVM_REGISTER_GLOBAL("Asinh").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::asinh);
});

TVM_REGISTER_GLOBAL("Acosh").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::acosh);
});

TVM_REGISTER_GLOBAL("LogicalNot").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::logical_not);
});

TVM_REGISTER_GLOBAL("LogicalAnd").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::logical_and);
});

TVM_REGISTER_GLOBAL("LogicalOr").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::logical_or);
});

TVM_REGISTER_GLOBAL("NotEqual").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::not_equal);
});

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

TVM_REGISTER_GLOBAL("IsNan").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::isnan);
});

TVM_REGISTER_GLOBAL("IsInf").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::isinf);
});

TVM_REGISTER_GLOBAL("IsFinite").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::isfinite);
});

TVM_REGISTER_GLOBAL("Tanh").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto call = [](const Tensor &tensor) {
    std::string name = "T_tanh_" + tensor->op->name;
    // Do not use topi::tanh here, since it expands the tanh with `fast_tanh_float` for fp32 tensors.
    return air::compute(
      tensor->shape, [&](const Array<Var> &i) { return air::tanh(tensor(i)); }, name, "elemwise");
  };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("TensorAdd").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::add);
});

TVM_REGISTER_GLOBAL("Add").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_TWO_INPUTS_CALL(args, rv, topi::add); });

TVM_REGISTER_GLOBAL("RealDiv").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::divide);
});

TVM_REGISTER_GLOBAL("Div").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::divide);
});

TVM_REGISTER_GLOBAL("FloorDiv").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::floor_divide);
});

TVM_REGISTER_GLOBAL("Mod").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_TWO_INPUTS_CALL(args, rv, topi::mod); });

TVM_REGISTER_GLOBAL("FloorMod").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::floor_mod);
});

TVM_REGISTER_GLOBAL("Floor").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::floor);
});

TVM_REGISTER_GLOBAL("Erf").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::erf); });

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

TVM_REGISTER_GLOBAL("Sin").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::sin); });

TVM_REGISTER_GLOBAL("Cos").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::cos); });

TVM_REGISTER_GLOBAL("Asin").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::asin); });

TVM_REGISTER_GLOBAL("ACos").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::acos); });

TVM_REGISTER_GLOBAL("Sign").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::sign); });

TVM_REGISTER_GLOBAL("ReduceSum").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  CHECK(attrs.count("keep_dims"));
  air::Array<air::Integer> axis = ArrayOrInt(attrs["axis"]);
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["keep_dims"])));
  auto call = [&axis, &keepdims](const air::Tensor &tensor) { return topi::sum(tensor, axis, keepdims); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("ReduceProd").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  CHECK(attrs.count("keep_dims"));
  air::Array<air::Integer> axis = ArrayOrInt(attrs["axis"]);
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["keep_dims"])));
  auto call = [&axis, &keepdims](const air::Tensor &tensor) { return topi::prod(tensor, axis, keepdims); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
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
  auto ref = [](OpAttr attrs) -> int {
    CHECK(attrs.count("axis"));
    auto axis = ir::GetInt32Const(Downcast<Array<Expr>>(attrs["axis"])[0]);
    return axis;
  };

  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, topi::expand_dims, ref);
});

TVM_REGISTER_GLOBAL("Reshape").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto ref = [](OpAttr attrs) -> Array<Expr> {
    CHECK(attrs.count("shape"));
    auto shape = Downcast<Array<Integer>>(attrs["shape"]);
    CHECK(!shape.empty());
    Array<Expr> newshape;
    for (auto s : shape) {
      newshape.push_back(s);
    }
    return newshape;
  };

  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, topi::reshape, ref);
});

TVM_REGISTER_GLOBAL("Transpose").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto ref = [](OpAttr attrs) -> Array<Integer> {
    CHECK(attrs.count("perm"));
    auto perm = Downcast<Array<Integer>>(attrs["perm"]);
    CHECK(!perm.empty());
    return perm;
  };
  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, topi::transpose, ref);
});

TVM_REGISTER_GLOBAL("Cast").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto type_mapping_copy = type_mapping;
  auto ref = [&type_mapping_copy](OpAttr attrs) -> Type {
    CHECK(attrs.count("dst_type"));
    auto attr = attrs["dst_type"];
    CHECK(attr->IsInstance<StringImm>());
    std::string dtype_str = attr.as<StringImm>()->value;
    if (type_mapping_copy.find(dtype_str) == type_mapping_copy.end()) {
      LOG(FATAL) << "Not support dtype: " << dtype_str;
    }
    return type_mapping_copy[dtype_str];
  };

  auto call = [](const Tensor &tensor, Type type) {
    std::string name = "T_cast_" + tensor->op->name;
    if (tensor->dtype == air::Float(32) && type == air::Bool()) {
      const char *runtime_mode = std::getenv("RUNTIME_MODE");
      if (runtime_mode == nullptr || (runtime_mode != nullptr && std::strstr(runtime_mode, "cloud") != nullptr)) {
        auto zero = make_zero(tensor->dtype);
        return topi::not_equal(tensor, zero);
      } else {
        auto tmp = topi::cast(tensor, air::Float(16), name + "tmp");
        auto zero = make_zero(air::Float(16));
        auto res = topi::not_equal(tmp, zero);
        return topi::cast(res, type, name);
      }
    }

    return topi::cast(tensor, type, name);
  };
  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, call, ref);
});

TVM_REGISTER_GLOBAL("Tile").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto ref = [](OpAttr attrs) -> Array<Integer> {
    CHECK(attrs.count("multiples"));
    auto multiples = Downcast<Array<Integer>>(attrs["multiples"]);
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
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  CHECK(attrs.count("keep_dims"));
  auto axis = ArrayOrInt(attrs["axis"]);
  CHECK(attrs["keep_dims"]->IsInstance<ExprNode>());
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["keep_dims"])));

  auto call = [&axis, &keepdims](const Tensor &tensor) { return topi::max(tensor, axis, keepdims); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("ReduceMin").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  CHECK(attrs.count("keep_dims"));
  auto axis = ArrayOrInt(attrs["axis"]);
  CHECK(attrs["keep_dims"]->IsInstance<ExprNode>());
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["keep_dims"])));

  auto call = [&axis, &keepdims](const Tensor &tensor) { return topi::min(tensor, axis, keepdims); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("Argmax").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  auto axis = ArrayOrInt(attrs["axis"]);

  auto inputs = args[0].operator Array<NodeRef>();
  auto data = Downcast<Tensor>(inputs[0]);

  bool reduce_on_single_element = false;
  if (axis.size() == 1 && data->shape.size() >= 1) {
    auto axis_size = (int64_t)Downcast<Integer>(data->shape[axis[0]]);
    if (axis_size == 1) {
      reduce_on_single_element = true;
    }
  }
  if (reduce_on_single_element) {
    size_t reduce_axis = (int64_t)axis[0];
    Array<Expr> reduce_shape;
    for (size_t i = 0; i < data->shape.size(); i++) {
      if (i != reduce_axis) reduce_shape.push_back(data->shape[i]);
    }
    *rv = compute(
      reduce_shape, [&](const Array<Var> &indices) { return make_const(data->dtype, 0); }, data->op->name + "_red",
      topi::kBroadcast);
  } else {
    auto call = [&axis](const Tensor &tensor) { return topi::argmax(tensor, axis, false); };
    TOPI_ONE_INPUT_CALL(args, rv, call);
  }
});

TVM_REGISTER_GLOBAL("Argmin").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  auto axis = ArrayOrInt(attrs["axis"]);

  auto inputs = args[0].operator Array<NodeRef>();
  auto data = Downcast<Tensor>(inputs[0]);

  bool reduce_on_single_element = false;
  if (axis.size() == 1 && data->shape.size() >= 1) {
    auto axis_size = (int64_t)Downcast<Integer>(data->shape[axis[0]]);
    if (axis_size == 1) {
      reduce_on_single_element = true;
    }
  }
  if (reduce_on_single_element) {
    size_t reduce_axis = (int64_t)axis[0];
    Array<Expr> reduce_shape;
    for (size_t i = 0; i < data->shape.size(); i++) {
      if (i != reduce_axis) reduce_shape.push_back(data->shape[i]);
    }
    *rv = compute(
      reduce_shape, [&](const Array<Var> &indices) { return make_const(data->dtype, 0); }, data->op->name + "_red",
      topi::kBroadcast);
  } else {
    auto call = [&axis](const Tensor &tensor) { return topi::argmin(tensor, axis, false); };
    TOPI_ONE_INPUT_CALL(args, rv, call);
  }
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

  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("depth"));
  CHECK(attrs.count("axis"));
  CHECK(attrs["depth"]->IsInstance<ExprNode>());
  CHECK(attrs["axis"]->IsInstance<ExprNode>());
  auto depth = ir::GetInt32Const(Downcast<Expr>(attrs["depth"]));
  auto axis = ir::GetInt32Const(Downcast<Expr>(attrs["axis"]));

  *rv = topi::one_hot(indices, on_value, off_value, depth, axis, indices->dtype);
});

TVM_REGISTER_GLOBAL("Reciprocal").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto call = [](const Tensor &tensor) { return topi::divide(make_const(tensor->dtype, 1.0), tensor); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
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
  auto condition = Downcast<Tensor>(inputs[0]);
  CHECK(inputs[1]->IsInstance<TensorNode>() || inputs[2]->IsInstance<TensorNode>());
  if (inputs[1]->IsInstance<TensorNode>() && inputs[2]->IsInstance<TensorNode>()) {
    auto x = Downcast<Tensor>(inputs[1]);
    auto y = Downcast<Tensor>(inputs[2]);
    *rv = topi::where(condition, x, y);
  } else if (inputs[1]->IsInstance<TensorNode>()) {
    auto x = Downcast<Tensor>(inputs[1]);
    auto shape = x->shape;
    auto y = compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(inputs[2]); });
    *rv = topi::where(condition, x, y);
  } else if (inputs[2]->IsInstance<TensorNode>()) {
    auto y = Downcast<Tensor>(inputs[2]);
    auto shape = y->shape;
    auto x = compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(inputs[1]); });
    *rv = topi::where(condition, x, y);
  }
});

TVM_REGISTER_GLOBAL("Equal").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::equal);
});

TVM_REGISTER_GLOBAL("Greater").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::greater);
});

TVM_REGISTER_GLOBAL("Less").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_TWO_INPUTS_CALL(args, rv, topi::less); });

TVM_REGISTER_GLOBAL("GreaterEqual").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::greater_equal);
});

TVM_REGISTER_GLOBAL("LessEqual").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::less_equal);
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

TVM_REGISTER_GLOBAL("Assign").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 2);
  bool in2_is_expr = inputs[1]->IsInstance<ExprNode>();
  bool in2_is_tensor = inputs[1]->IsInstance<TensorNode>();
  CHECK(inputs[0]->IsInstance<TensorNode>()) << "Input1 should be of type Tensor";
  CHECK(in2_is_expr || in2_is_tensor) << "Input2 should be of type Expr or Tensor";
  auto ref = Downcast<Tensor>(inputs[0]);
  auto val = in2_is_expr
               ? compute(ref->shape, [&](const Array<Var> &indices) { return Downcast<Expr>(inputs[1]); })
               : compute(ref->shape, [&](const Array<Var> &indices) { return Downcast<Tensor>(inputs[1])(indices); });
  *rv = val;
});

TVM_REGISTER_GLOBAL("EquivFormat").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  if (inputs[0]->IsInstance<TensorNode>()) {
    auto ref = [](OpAttr attrs) -> Array<Expr> {
      CHECK(attrs.count("shape"));
      auto shape = Downcast<Array<Integer>>(attrs["shape"]);
      CHECK(!shape.empty());
      Array<Expr> newshape;
      for (auto s : shape) {
        newshape.push_back(s);
      }
      return newshape;
    };

    TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, topi::reshape, ref);
  } else {
    Array<Expr> shape = {Expr(1)};
    *rv = compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(inputs[0]); });
  }
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

TVM_REGISTER_GLOBAL("BroadcastTo").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_EQ(args.size(), 2);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_EQ(inputs.size(), 1);
  auto attrs = args[1].operator OpAttr();
  auto shape_v = Downcast<Array<Integer>>(attrs["shape"]);
  Array<Expr> shape;
  for (size_t i = 0; i < shape_v.size(); ++i) {
    shape.push_back(shape_v[i]);
  }
  if (inputs[0]->IsInstance<ExprNode>()) {
    auto val = Downcast<Expr>(inputs[0]);
    auto fcompute = [&](const Array<Var> &indices) { return val; };
    *rv = compute(shape, fcompute, "broadcast");
  } else {
    auto val = Downcast<Tensor>(inputs[0]);
    *rv = topi::broadcast_to(val, shape);
  }
});

TVM_REGISTER_GLOBAL("CudaBatchMatMul").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto inputs = args[0].operator Array<NodeRef>();
  auto attrs = args[1].operator OpAttr();
  CHECK_GE(inputs.size(), 2);
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  auto left_matrix = Downcast<Tensor>(inputs[0]);
  auto right_matrix = Downcast<Tensor>(inputs[1]);
  CHECK(attrs.count("transpose_a"));
  CHECK(attrs.count("transpose_b"));
  CHECK(attrs.count("dst_type"));
  auto dst_type = GetString(attrs["dst_type"]);
  bool transpose_a = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["transpose_a"])));
  bool transpose_b = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["transpose_b"])));
  auto left_shape = left_matrix->shape;
  auto right_shape = right_matrix->shape;
  CHECK_EQ(left_shape.size(), right_shape.size());

  auto type_checker = [](const Tensor &input_data, const std::string name) {
    if (input_data->dtype != Float(16)) {
      LOG(FATAL) << "dtype of input tensor " << name << " should be float16";
    }
  };

  Expr k;
  auto compute_out = [&k](const Array<Expr> &left_shape, const Array<Expr> &right_shape, bool transpose_a,
                          bool transpose_b, size_t batch_dim) {
    auto m = left_shape[batch_dim];
    k = left_shape[batch_dim + 1];
    if (transpose_a) {
      m = left_shape[batch_dim + 1];
      k = left_shape[batch_dim];
    }
    auto n = right_shape[batch_dim + 1];
    if (transpose_b) {
      n = right_shape[batch_dim];
    }
    Array<Expr> output_shape;
    for (size_t i = 0; i < batch_dim; ++i) {
      output_shape.push_back(left_shape[i]);
    }
    output_shape.push_back(m);
    output_shape.push_back(n);
    return output_shape;
  };

  size_t batch_dim = 0;
  IterVar reduce_k;
  auto fcompute = [&left_matrix, &right_matrix, &transpose_a, &transpose_b, &reduce_k, &batch_dim,
                   &dst_type](const Array<Var> &indices) {
    Array<Expr> left_indice;
    Array<Expr> right_indice;
    for (size_t i = 0; i < batch_dim; ++i) {
      left_indice.push_back(indices[i]);
      right_indice.push_back(indices[i]);
    }

    if (transpose_a) {
      left_indice.push_back(reduce_k);
      left_indice.push_back(indices[batch_dim]);
    } else {
      left_indice.push_back(indices[batch_dim]);
      left_indice.push_back(reduce_k);
    }

    if (transpose_b) {
      right_indice.push_back(indices[batch_dim + 1]);
      right_indice.push_back(reduce_k);
    } else {
      right_indice.push_back(reduce_k);
      right_indice.push_back(indices[batch_dim + 1]);
    }

    Expr left_buffer = Call::make(left_matrix->dtype, left_matrix->op->name, left_indice, Call::CallType::Halide,
                                  left_matrix->op, left_matrix->value_index);
    Expr right_buffer = Call::make(right_matrix->dtype, right_matrix->op->name, right_indice, Call::CallType::Halide,
                                   right_matrix->op, right_matrix->value_index);

    if (dst_type == "float32") {
      left_buffer = Cast::make(Float(32), left_buffer);
      right_buffer = Cast::make(Float(32), right_buffer);
    }

    auto matrix_mul = Mul::make(left_buffer, right_buffer);
    Array<IterVar> reduces;
    reduces.push_back(reduce_k);
    auto res = air::sum(matrix_mul, reduces);
    return res;
  };

  type_checker(left_matrix, "left_matrix");
  type_checker(right_matrix, "right_matrix");
  batch_dim = left_shape.size() - 2;
  Array<Expr> output_shape = compute_out(left_shape, right_shape, transpose_a, transpose_b, batch_dim);
  reduce_k = air::reduce_axis(Range(0, k), "reduce_axis");
  auto name = "T_batch_matmul_" + left_matrix->op->name + "_" + right_matrix->op->name;
  *rv = compute(output_shape, fcompute, name, "matmul");
});

// only support fractal_zN: [ko mo mi ki] * [no ko ki ni] = [no mo mi ni]
void AicoreCubeMatMul(const TVMArgs &args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("transpose_a"));
  CHECK(attrs.count("transpose_b"));
  CHECK(attrs.count("dst_type"));
  CHECK(attrs.count("left_format"));
  CHECK(attrs.count("right_format"));
  bool transpose_a = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["transpose_a"])));
  bool transpose_b = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["transpose_b"])));
  auto dst_type = GetString(attrs["dst_type"]);
  auto left_format = GetString(attrs["left_format"]);
  auto right_format = GetString(attrs["right_format"]);
  if (right_format != "FRACTAL_NZ" || left_format != "FRACTAL_NZ") {
    LOG(FATAL) << "format of " << left_format << "*" << right_format << " is not supported";
  }

  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  auto left_matrix = Downcast<Tensor>(inputs[0]);
  auto right_matrix = Downcast<Tensor>(inputs[1]);
  auto left_shape = left_matrix->shape;
  auto right_shape = right_matrix->shape;
  CHECK_EQ(left_shape.size(), right_shape.size());
  CHECK_GE(left_shape.size(), 4);

  auto type_checker = [](const Tensor &input_data, const std::string name, const air::DataType type) {
    if (input_data->dtype != type) {
      LOG(FATAL) << "dtype of " << name << " is not supported";
    }
  };
  type_checker(left_matrix, "left_matrix", Float(16));
  type_checker(right_matrix, "right_matrix", Float(16));

  // compute m n k
  Array<Expr> output_shape;
  Array<Expr> k;
  auto compute_mnk = [&output_shape, &k, &left_shape, &right_shape, transpose_a, transpose_b]() {
    size_t dim = left_shape.size();
    Expr mo, mi, no, ni, ko, ki;
    if (transpose_a) {
      mo = left_shape[dim - 4];
      ko = left_shape[dim - 3];
      ki = left_shape[dim - 2];
      mi = left_shape[dim - 1];
    } else {
      ko = left_shape[dim - 4];
      mo = left_shape[dim - 3];
      mi = left_shape[dim - 2];
      ki = left_shape[dim - 1];
    }
    if (transpose_b) {
      no = right_shape[dim - 3];
      ni = right_shape[dim - 2];
    } else {
      no = right_shape[dim - 4];
      ni = right_shape[dim - 1];
    }
    for (size_t i = 0; i < dim - 4; ++i) {
      output_shape.push_back(left_shape[i]);
    }
    output_shape.push_back(no);
    output_shape.push_back(mo);
    output_shape.push_back(mi);
    output_shape.push_back(ni);
    k = {ko, ki};
  };

  compute_mnk();

  // define fcompute
  auto Mmad = [](Expr source, const Array<IterVar> &rdom) {
    Var x("x", source.type()), y("y", source.type());
    Expr result = Call::make(source.type(), "mad", {x, y}, Call::PureIntrinsic);
    Expr identity_element = make_zero(source.type());
    CommReducer combiner = CommReducerNode::make({x}, {y}, {result}, {identity_element});
    return Reduce::make(combiner, {source}, rdom, make_const(Bool(1), true), 0);
  };

  IterVar reduce_ko = air::reduce_axis(Range(0, k[0]), "ko");
  IterVar reduce_ki = air::reduce_axis(Range(0, k[1]), "ki");
  Array<IterVar> reduces = {reduce_ko, reduce_ki};

  auto fcompute = [&left_matrix, &right_matrix, &transpose_a, &transpose_b, &reduces,
                   &Mmad](const Array<Var> &indices) {
    size_t dim = indices.size();
    Array<Expr> left_indice;
    for (size_t i = 0; i < dim - 4; ++i) {
      left_indice.push_back(indices[i]);
    }
    if (transpose_a) {
      left_indice.push_back(indices[dim - 3]);
      left_indice.push_back(reduces[0]);
      left_indice.push_back(reduces[1]);
      left_indice.push_back(indices[dim - 2]);
    } else {
      left_indice.push_back(reduces[0]);
      left_indice.push_back(indices[dim - 3]);
      left_indice.push_back(indices[dim - 2]);
      left_indice.push_back(reduces[1]);
    }

    Array<Expr> right_indice;
    for (size_t i = 0; i < dim - 4; ++i) {
      right_indice.push_back(indices[i]);
    }
    if (transpose_b) {
      right_indice.push_back(reduces[0]);
      right_indice.push_back(indices[dim - 4]);
      right_indice.push_back(indices[dim - 1]);
      right_indice.push_back(reduces[1]);
    } else {
      right_indice.push_back(indices[dim - 4]);
      right_indice.push_back(reduces[0]);
      right_indice.push_back(reduces[1]);
      right_indice.push_back(indices[dim - 1]);
    }

    Expr res = Mmad(Cast::make(Float(32), left_matrix(left_indice) * right_matrix(right_indice)), reduces);
    return res;
  };

  // set output name
  auto name = "T_batchmatmul_" + left_matrix->op->name + "_" + right_matrix->op->name;

  // set compute attrs
  auto set_compute_attrs_zN = [&left_matrix, &right_matrix, &inputs, &output_shape, &dst_type, &k, transpose_a, transpose_b, attrs]() {
    Map<std::string, NodeRef> com_attrs;

    com_attrs.Set("pragma_gemm_output_shape", output_shape);
    com_attrs.Set("pragma_gemm_k", k);
    com_attrs.Set("pragma_gemm_data", Expr(left_matrix->op->name));
    com_attrs.Set("pragma_gemm_weight", Expr(right_matrix->op->name));
    com_attrs.Set("pragma_conv_bypass_l1", Expr(0));
    if (attrs.count("bypass")) {
      com_attrs.Set("pragma_conv_bypass_l1", Downcast<Expr>(attrs["bypass"]));
    }

    std::string data_trans("Y");
    std::string data_trans_block("Y");
    std::string data_trans_block_in("N");
    if (transpose_a) {
      data_trans = "Y";
      data_trans_block = "N";
      data_trans_block_in = "Y";
    }
    com_attrs.Set("pragma_data_transpose", Expr(data_trans));
    com_attrs.Set("pragma_data_transpose_block", Expr(data_trans_block));
    com_attrs.Set("pragma_data_transpose_block_inner", Expr(data_trans_block_in));

    std::string weight_trans("Y");
    std::string weight_trans_block("N");
    std::string weight_trans_block_in("N");
    if (transpose_b) {
      weight_trans = "N";
      weight_trans_block = "N";
      weight_trans_block_in = "N";
    }
    com_attrs.Set("pragma_weight_transpose", Expr(weight_trans));
    com_attrs.Set("pragma_weight_transpose_block", Expr(weight_trans_block));
    com_attrs.Set("pragma_weight_transpose_block_inner", Expr(weight_trans_block_in));

    com_attrs.Set("bias", Expr(""));
    if (inputs.size() > 2) {
      CHECK(inputs[2]->IsInstance<TensorNode>());
      auto bias = Downcast<Tensor>(inputs[2]);
      com_attrs.Set("bias", Expr(bias->op->name));
    }

    return com_attrs;
  };

  auto com_attrs = set_compute_attrs_zN();

  // compute matmul(a,b)
  auto c_tensor = compute(output_shape, fcompute, name, "matmul", com_attrs);

  if (inputs.size() > 2) {
    auto bias = Downcast<Tensor>(inputs[2]);
    if (bias->dtype == Float(16)) {
      bias = topi::cast(bias, Float(32));
    } else {
      type_checker(bias, "bias", Float(32));
    }
    c_tensor = topi::add(c_tensor, bias);
  }

  if (dst_type == "float16") {
    c_tensor = topi::cast(c_tensor, Float(16));
  } else {
    type_checker(c_tensor, "dst_type", Float(32));
  }

  *rv = c_tensor;
}

void AicoreVectorMatMul(const TVMArgs &args, TVMRetValue *rv) {
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("transpose_a"));
  CHECK(attrs.count("transpose_b"));
  bool transpose_a = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["transpose_a"])));
  bool transpose_b = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs["transpose_b"])));
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 2);
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  auto left_matrix = Downcast<Tensor>(inputs[0]);
  auto right_matrix = Downcast<Tensor>(inputs[1]);
  auto left_shape = left_matrix->shape;
  auto right_shape = right_matrix->shape;
  CHECK_EQ(left_shape.size(), right_shape.size());
  CHECK_GE(left_shape.size(), 2);

  // For the matmul, if use fp16 to accumulate, there will be some precision problems.
  // Therefore, for the VectorMatMul, the input needs to be casted to fp32.
  auto dtype = left_matrix->dtype;
  if (dtype == Float(16)) {
    left_matrix = topi::cast(left_matrix, Float(32));
    right_matrix = topi::cast(right_matrix, Float(32));
  }

  // compute m n k
  Array<Expr> output_shape;
  Expr k;
  auto compute_mnk = [&output_shape, &k, &left_shape, &right_shape, transpose_a, transpose_b]() {
    size_t dim = left_shape.size();
    Expr m, n;
    if (transpose_a) {
      k = left_shape[dim - 2];
      m = left_shape[dim - 1];
    } else {
      m = left_shape[dim - 2];
      k = left_shape[dim - 1];
    }
    if (transpose_b) {
      n = right_shape[dim - 2];
    } else {
      n = right_shape[dim - 1];
    }
    for (size_t i = 0; i < dim - 2; ++i) {
      output_shape.push_back(left_shape[i]);
    }
    output_shape.push_back(m);
    output_shape.push_back(n);
  };

  compute_mnk();

  // define fcompute
  IterVar reduce_k = air::reduce_axis(Range(0, k), "k");

  auto fcompute = [&left_matrix, &right_matrix, &transpose_a, &transpose_b, &reduce_k](const Array<Var> &indices) {
    size_t dim = indices.size();
    Array<Expr> left_indice;
    for (size_t i = 0; i < dim - 2; ++i) {
      left_indice.push_back(indices[i]);
    }
    if (transpose_a) {
      left_indice.push_back(reduce_k);
      left_indice.push_back(indices[dim - 2]);
    } else {
      left_indice.push_back(indices[dim - 2]);
      left_indice.push_back(reduce_k);
    }

    Array<Expr> right_indice;
    for (size_t i = 0; i < dim - 2; ++i) {
      right_indice.push_back(indices[i]);
    }
    if (transpose_b) {
      right_indice.push_back(indices[dim - 1]);
      right_indice.push_back(reduce_k);
    } else {
      right_indice.push_back(reduce_k);
      right_indice.push_back(indices[dim - 1]);
    }

    Expr res = air::sum(left_matrix(left_indice) * right_matrix(right_indice), {reduce_k});
    return res;
  };

  // set output name
  auto name = "T_batchmatmul_" + left_matrix->op->name + "_" + right_matrix->op->name;

  // compute matmul(a,b)
  auto c_tensor = compute(output_shape, fcompute, name, "matmul");

  CHECK(attrs.count("dst_type"));
  auto dst_type = GetString(attrs["dst_type"]);
  if (dst_type == "float16") {
    c_tensor = topi::cast(c_tensor, Float(16));
  }

  // bias add
  if (inputs.size() > 2) {
    auto bias = Downcast<Tensor>(inputs[2]);
    if (bias->dtype != c_tensor->dtype) {
      bias = topi::cast(bias, c_tensor->dtype);
    }
    c_tensor = topi::add(c_tensor, bias);
  }

  *rv = c_tensor;
}

TVM_REGISTER_GLOBAL("AicoreBatchMatMul").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("left_format"));
  CHECK(attrs.count("right_format"));
  auto left_format = GetString(attrs["left_format"]);
  auto right_format = GetString(attrs["right_format"]);

  auto is_default_format = [](const std::string &format_name) -> bool {
    if (format_name == "DefaultFormat" || format_name == "NCHW") {
      return true;
    }
    return false;
  };

  if (left_format == "FRACTAL_NZ" && right_format == "FRACTAL_NZ") {
    return AicoreCubeMatMul(args, rv);
  } else if (is_default_format(left_format) && is_default_format(right_format)) {
    return AicoreVectorMatMul(args, rv);
  } else {
    LOG(FATAL) << "format of " << left_format << "*" << right_format << " is not supported";
  }
});

#ifdef ENABLE_GENERAL_TOT
TVM_REGISTER_GLOBAL("Gather").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  auto input0 = Downcast<Tensor>(inputs[0]);
  auto input1 = Downcast<Tensor>(inputs[1]);
  auto attrs = args[1].operator OpAttr();
  CHECK(attrs.count("axis"));
  auto axis = (size_t)ir::GetInt32Const(Downcast<Array<Expr>>(attrs["axis"])[0]);
  auto x_shape = input0->shape;
  auto y_shape = input1->shape;
  CHECK(y_shape.size() == 1);
  Map<std::string, NodeRef> com_attrs;
  com_attrs.Set("no_inline", Expr(-1)); // out tensor is not inlined
  // used for replace_tot, out = gather(par, index)
  com_attrs.Set("tensor_of_tensor_pos", Expr(0)); // to mark tensor_not_promote
  com_attrs.Set("first_index_pos", Expr(1)); // to mark inner_tensor
  // used for recover_tot, out = tot_op(par, index)
  com_attrs.Set("is_fakeout", Expr(0)); // to remove fakeout
  com_attrs.Set("realout_pos", Expr(-1)); // -1 means provide func
  com_attrs.Set("first_dst_pos", Expr((int)axis)); // which axis in dst_tenosr
  com_attrs.Set("outbound_return_zero", Expr(1)); // need else stmt
  com_attrs.Set("is_atomic_add", Expr(0));
  Array<Expr> output_shape;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (i == axis) {
      output_shape.push_back(y_shape[0]);
    } else {
      output_shape.push_back(x_shape[i]);
    }
  }
  auto fcompute = [&input0, &input1, &axis](const Array<Var> &indices) {
    Array<Expr> x_shape;
    Array<Expr> y_shape;
    for (size_t i = 0; i < indices.size(); ++i) {
      x_shape.push_back(indices[i]);
      if (axis == i) {
        y_shape.push_back(indices[i]);
      }
    }
    Expr x = Call::make(input0->dtype, input0->op->name, x_shape, Call::CallType::Halide, input0->op);
    Expr y = Call::make(input1->dtype, input1->op->name, y_shape, Call::CallType::Halide, input1->op);
    Expr gather_call = Call::make(input0->dtype, "Gather", {x, y}, Call::PureIntrinsic);
    return gather_call;
  };

  std::string name = "T_gather_" + input0->op->name + "_" + input1->op->name + "_" + std::to_string(axis);
  auto res_tensor = compute(output_shape, fcompute, name, "tot", com_attrs);
  *rv = res_tensor;
});

TVM_REGISTER_GLOBAL("TensorScatterAdd").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  CHECK(inputs[2]->IsInstance<TensorNode>());
  auto input0 = Downcast<Tensor>(inputs[0]);
  auto input1 = Downcast<Tensor>(inputs[1]);
  auto input2 = Downcast<Tensor>(inputs[2]);
  auto par_shape = input0->shape;
  auto index_shape = input1->shape;
  auto update_shape = input2->shape;
  int depth = 1;
  auto index_rank = index_shape.size();
  if (index_rank > 1) {
    depth = ir::GetInt32Const(index_shape[index_rank - 1]);
  }
  /* the axis for par_batch is unused after replace_tot. Reshape is
  needed if par_batch is < 0
  */
  size_t par_batch = par_shape.size() - update_shape.size();
  CHECK(par_batch >= 0);
  // output_shape should comes from real axis
  auto output_shape = par_shape;
  for (size_t i = par_batch; i < par_shape.size(); ++i) {
    output_shape.Set(i - par_batch, update_shape[i]);
  }
  Map<std::string, NodeRef> com_attrs;
  com_attrs.Set("no_inline", Expr(1)); // up is not inlined
  // used for replace_tot, fakeout = tsa(par, up, index)
  com_attrs.Set("tensor_of_tensor_pos", Expr(0)); // to mark tensor_not_promote
  com_attrs.Set("first_index_pos", Expr(2)); // to mark inner_tensor
  // used for recover_tot, fakeout = tot_op(par, up, index)
  com_attrs.Set("is_fakeout", Expr(1)); // to remove fakeout
  com_attrs.Set("realout_pos", Expr(0));
  com_attrs.Set("first_dst_pos", Expr(0)); // which axis in tensor_of_tensor
  com_attrs.Set("outbound_return_zero", Expr(0)); // no else stmt
  com_attrs.Set("is_atomic_add", Expr(1));

  auto fcompute = [&index_rank, &depth, &par_batch, &input0, &input1, &input2](const Array<Var> &indices) {
    Array<Expr> output_index;
    for (size_t i = 0; i < indices.size(); ++i) {
      output_index.push_back(indices[i]);
    }
    Expr par = Call::make(input0->dtype, input0->op->name, output_index, Call::CallType::Halide, input0->op);

    Array<Expr> update_index;
    for (size_t i = par_batch; i < indices.size(); ++i) {
      update_index.push_back(indices[i]);
    }
    Expr update = Call::make(input2->dtype, input2->op->name, update_index, Call::CallType::Halide, input2->op);

    Array<Expr> index_index{update_index[0]};
    if (index_rank > 2) {
      for (size_t i = 1; i < index_rank - 1; ++i) {
        index_index.push_back(update_index[i]);
      }
    }
    Array<Expr> args{par, update};
    if (index_rank > 1) {
      for (int i = 0; i < depth; ++i) {
        auto index_full = index_index;
        index_full.push_back(Expr(i));
        auto index = Call::make(input1->dtype, input1->op->name, index_full, Call::CallType::Halide, input1->op);
        args.push_back(index);
      }
      Expr tsa_call = Call::make(input0->dtype, "TensorScatterAdd", args, Call::PureIntrinsic);
      return tsa_call;
    }

    Expr index = Call::make(input1->dtype, input1->op->name, index_index, Call::CallType::Halide, input1->op);
    args.push_back(index);
    Expr tsa_call = Call::make(input0->dtype, "TensorScatterAdd", args, Call::PureIntrinsic);
    return tsa_call;
  };
  std::string name = "tensor_scatter_add";
  auto res_tensor = compute(output_shape, fcompute, name, "tot", com_attrs);
  *rv = res_tensor;
});
#endif

TVM_REGISTER_GLOBAL("Atan").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::atan); });

TVM_REGISTER_GLOBAL("Atan2").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_EQ(inputs.size(), 2);
  Tensor x_tensor = Downcast<Tensor>(inputs[0]);
  Tensor y_tensor = Downcast<Tensor>(inputs[1]);
  *rv = topi::atan2(x_tensor, y_tensor);
});

TVM_REGISTER_GLOBAL("Expm1").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::expm1);
});

}  // namespace akg
