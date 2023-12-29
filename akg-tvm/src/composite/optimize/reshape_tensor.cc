/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "composite/optimize/pass.h"
#include <stack>

namespace akg {
constexpr int OFFSET1 = 1;
constexpr int OFFSET2 = 2;
constexpr int OFFSET3 = 3;
constexpr int OFFSET4 = 4;

/* Use reshape to deal with FRACTAL_NZ and DefaultFormat in element-wise and broadcast supported operators. e.g.
 *
 *  output_0_1(256, 256, 16, 16) = TensorAdd(input_1(256, 256, 16, 16), output_0_0(4096))
 *  ------------>
 *  output_0_0_reshape(256, 1, 1, 16) = Reshape(output_0_0(4096))
 *  output_0_1(256, 256, 16, 16) = TensorAdd(input_1(256, 256, 16, 16), output_0_0_reshape(256, 1, 1, 16))
 *
 *  where input_1 is FRACTAL_NZ, output_0_0 is DefaultFormat
 */
class ReshapeTensorMutator : public IRMutator {
 public:
  explicit ReshapeTensorMutator() = default;
  ~ReshapeTensorMutator() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    attr_level_++;
    Map<std::string, NodeRef> attr_map;
    // 1. Gather attribute map
    GatherAttrs(op, attr_map);

    // 2. Traverse current AttrStmt
    auto stmt = this->Mutate(op->body);

    // 3. Modify the attribute map if needed
    stmt = ModifyAttrMap(op, stmt, attr_map);

    // 4. Add reshape stmt before the top-level AttrStmt if needed
    return AddReshape(stmt);
  }

  virtual bool CheckList(const std::string &name) {
    static std::unordered_set<std::string> check_list = {"TensorAdd", "Add",     "RealDiv", "Mul",
                                                         "Minimum",   "Maximum", "Sub"};
    return check_list.find(name) != check_list.end();
  }

  virtual bool CheckArgs(const Array<Expr> &args) {
    CHECK_EQ(args.size(), 2);
    return false;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto call = op->value.as<Call>();
    if (call == nullptr || !CheckList(call->name)) {
      return IRMutator::Mutate_(op, s);
    }

    auto args = call->args;
    if (CheckArgs(args)) {
      return IRMutator::Mutate_(op, s);
    }

    Array<Expr> shape_fractal;
    Array<Expr> shape_default;
    bool need_swap = false;
    if (!GetShapes(call, shape_fractal, shape_default, need_swap)) {
      return IRMutator::Mutate_(op, s);
    }

    Array<Expr> shape0;
    Array<Expr> shape1;
    Array<Expr> shape_out;
    auto orig_shape = InferShapeFromFractalNz(shape_fractal);
    std::tie(shape0, shape1, shape_out) = ProduceShapes(orig_shape, shape_default);
    auto op_name = call->name;
    auto shape_new = InferShapeToFractalNz(shape0, shape1, shape_out, shape_fractal, op_name, shape_default);

    // Rewrite current provide stmt's input shape
    if (!EqualShape(shape_default, shape_new)) {
      LOG(INFO) << "[" << op_name << "] " << shape_default << " will be reshaped to " << shape_new;
      auto input_tensors = CollectInputTensors(need_swap, args, shape_new);
      return ModifyProvideInput(input_tensors, op);
    }

    return IRMutator::Mutate_(op, s);
  }

  bool GetFormats(const std::string &op_name, const Tensor &tensor0, const Tensor &tensor1, std::string &format0,
                  std::string &format1) {
    if (!tensor0.defined() || !tensor1.defined()) {
      LOG(INFO) << "[" << op_name << "] not all inputs are tensors";
      return false;
    }
    LOG(INFO) << "[" << op_name << "] input shapes: " << tensor0->shape << ", " << tensor1->shape;

    if (!tensor0->op.defined() || !tensor1->op.defined()) {
      LOG(INFO) << "[" << op_name << "] not all input tensors have operation";
      return false;
    }

    format0 = GetDataFormat(tensor0->op->name);
    format1 = GetDataFormat(tensor1->op->name);
    LOG(INFO) << "[" << op_name << "] input formats: " << format0 << ", " << format1;
    if (format0 == format1) {
      return false;
    }
    return true;
  }

  virtual bool GetShapes(const Call *call, Array<Expr> &shape_fractal, Array<Expr> &shape_default, bool &need_swap) {
    auto args = call->args;
    auto op_name = call->name;

    Tensor tensor0 = RecoverTensor(args[0]);
    Tensor tensor1 = RecoverTensor(args[1]);

    std::string format0;
    std::string format1;
    if (!GetFormats(op_name, tensor0, tensor1, format0, format1)) {
      return false;
    }
    if (format0 == "FRACTAL_NZ" && format1 == "DefaultFormat") {
      shape_fractal = tensor0->shape;
      shape_default = tensor1->shape;
    } else if (format0 == "DefaultFormat" && format1 == "FRACTAL_NZ") {
      shape_fractal = tensor1->shape;
      shape_default = tensor0->shape;
      need_swap = true;
    } else {
      LOG(WARNING) << "[" << op_name << "], check whether input formats are valid";
      return false;
    }

    return true;
  }

  virtual Array<Tensor> CollectInputTensors(bool need_swap, const Array<Expr> &args, const Array<Expr> &shape_new) {
    Array<Tensor> input_tensors;
    Tensor tensor0 = RecoverTensor(args[0]);
    Tensor tensor1 = RecoverTensor(args[1]);
    if (need_swap) {
      auto reshape_tensor = placeholder(shape_new, tensor0->dtype, tensor0->op->name + "_reshape");
      reshape_.Set(tensor0, reshape_tensor);
      input_tensors.push_back(reshape_tensor);
      input_tensors.push_back(tensor1);
    } else {
      auto reshape_tensor = placeholder(shape_new, tensor1->dtype, tensor1->op->name + "_reshape");
      reshape_.Set(tensor1, reshape_tensor);
      input_tensors.push_back(tensor0);
      input_tensors.push_back(reshape_tensor);
    }
    return input_tensors;
  }

  virtual Array<Expr> InferShapeToFractalNz(const Array<Expr> &shape0, const Array<Expr> &shape1,
                                            const Array<Expr> &shape_out, const Array<Expr> &shape_fractal,
                                            const std::string &op_name, const Array<Expr> &shape_default) {
    auto dims = shape_out.size();
    auto batch = dims - OFFSET2;
    Array<Expr> shape_new;
    CHECK(dims >= 2);

    CHECK(shape0.size() == shape1.size());
    CHECK(shape1.size() == shape_out.size());

    for (size_t i = 0; i < batch; ++i) {
      shape_new.push_back(shape1[i]);
    }
    if (is_one(shape1[dims - OFFSET2]) && is_one(shape1[dims - OFFSET1])) {
      // (bs, 1, 1) --> (bs, 1, 1, 1, 1)
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
    } else if (is_one(shape1[dims - OFFSET2]) && GetDim(shape1, dims - OFFSET1) == GetDim(shape0, dims - OFFSET1)) {
      // (bs, 1, n) --> (bs, n1, 1, 1, n0), where n = n1 * n0
      shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET4]);
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
      shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET1]);
    } else if (GetDim(shape1, dims - OFFSET2) == GetDim(shape0, dims - OFFSET2) && is_one(shape1[dims - OFFSET1])) {
      // (bs, m, 1) --> (bs, 1, m1, m0, 1), where m = m1 * m0
      shape_new.push_back(Expr(1));
      shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET3]);
      shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET2]);
      shape_new.push_back(Expr(1));
    } else {
      // (bs, m, n), in this case, data format transformation is needed
      LOG(FATAL) << "[" << op_name << "] " << shape_fractal << " (FRACTAL_NZ) and " << shape_default
                 << " (DefaultFormat) may need data format transformation for ";
    }
    return shape_new;
  }

  virtual Array<Expr> InferShapeFromFractalNz(const Array<Expr> &fractal) {
    CHECK(fractal.size() >= 4);

    Array<Expr> shape;
    size_t dims = fractal.size();
    size_t batch = dims - OFFSET4;
    for (size_t i = 0; i < batch; ++i) {
      shape.push_back(fractal[i]);
    }
    int64_t m = GetDim(fractal, dims - OFFSET3) * GetDim(fractal, dims - OFFSET2);
    int64_t n = GetDim(fractal, dims - OFFSET4) * GetDim(fractal, dims - OFFSET1);
    shape.push_back(Expr(m));
    shape.push_back(Expr(n));

    return shape;
  }

 protected:
  int attr_level_{0};
  Map<std::string, NodeRef> op_attrs_;
  Map<Tensor, Tensor> reshape_;

  Stmt ModifyProvideInput(const Array<Tensor> &input_tensors, const Provide *op) {
    Array<Expr> input;
    for (const auto &ele : input_tensors) {
      input.push_back(Call::make(ele->dtype, ele->op->name, ele->shape, Call::CallType::Halide, ele->op));
    }
    auto call = op->value.as<Call>();
    return Provide::make(op->func, 0, Call::make(op->value.type(), call->name, input, Call::CallType::PureIntrinsic),
                         op->args);
  }

  Stmt ModifyAttrMap(const AttrStmt *op, const Stmt &stmt, const Map<std::string, NodeRef> &attr_map) {
    bool update_attr_map = false;
    Map<std::string, NodeRef> attr_map_new;

    if (!reshape_.empty() && attr_map.defined()) {
      std::unordered_map<std::string, std::string> format_key;
      for (const auto &it : reshape_) {
        auto key_f = CreateDataFormatKey(it.first->op->name);
        auto key_s = CreateDataFormatKey(it.second->op->name);
        format_key[key_f] = key_s;
      }

      for (const auto &it : attr_map) {
        if (format_key.find(it.first) != format_key.end()) {
          update_attr_map = true;
          attr_map_new.Set(format_key[it.first], StringImm::make("FRACTAL_NZ"));
        } else {
          attr_map_new.Set(it.first, it.second);
        }
      }
    }

    if (update_attr_map) {
      return AttrStmt::make(attr_map_new, op->attr_key, op->value, stmt);
    }
    return AttrStmt::make(op->node, op->attr_key, op->value, stmt);
  }

  Stmt AddReshape(Stmt &stmt) {
    if (attr_level_ != 1) {
      return stmt;
    }

    if (!reshape_.empty()) {
      for (const auto &it : reshape_) {
        auto arg =
          Call::make(it.first->dtype, it.first->op->name, it.first->shape, Call::CallType::Halide, it.first->op);
        auto reshape_stmt =
          Provide::make(it.second->op, 0, Call::make(it.first->dtype, "Reshape", {arg}, Call::CallType::PureIntrinsic),
                        it.second->shape);
        Map<std::string, NodeRef> attrs;
        attrs.Set("shape", it.second->shape);
        auto reshape_attr = AttrStmt::make(attrs, "attrs", Expr(1), reshape_stmt);
        stmt = Block::make(reshape_attr, stmt);
      }
    }

    // Reset status of current top-level AttrStmt
    attr_level_ = 0;
    op_attrs_ = {};
    reshape_ = {};

    return stmt;
  }

  void GatherAttrs(const AttrStmt *op, Map<std::string, NodeRef> &attr_map) {
    if (!op->node.as<StrMapNode>()) {
      return;
    }
    attr_map = Downcast<Map<std::string, NodeRef>>(op->node);
    for (const auto &it : attr_map) {
      if (op_attrs_.find(it.first) != op_attrs_.end()) {
        LOG(WARNING) << it.first << " already exist in attribute map, original value: " << op_attrs_[it.first]
                     << ", overridden value: " << it.second;
      }
      op_attrs_.Set(it.first, it.second);
    }
  }

  Tensor RecoverTensor(const Expr &e) {
    Tensor ret;
    auto call = e.as<Call>();
    if (call == nullptr || call->call_type != Call::CallType::Halide) {
      return ret;
    }

    auto func = call->func;
    if (!func.defined() || func.as<OperationNode>() == nullptr) {
      return ret;
    }

    ret = Downcast<Operation>(func).output(0);
    return ret;
  }

  std::string GetDataFormat(const std::string &name) {
    std::string format;
    auto key = CreateDataFormatKey(name);
    if (op_attrs_.find(key) != op_attrs_.end()) {
      auto value = op_attrs_[key].as<StringImm>();
      if (value != nullptr) {
        format = value->value;
      }
    }
    return format;
  }

  static int64_t GetDim(const Array<Expr> &shape, size_t idx) {
    CHECK(idx >= 0 && idx < shape.size());
    auto dim = shape[idx].as<IntImm>();
    CHECK(dim != nullptr);
    return dim->value;
  }

  std::tuple<Array<Expr>, Array<Expr>, Array<Expr>> ProduceShapes(const Array<Expr> &shape1,
                                                                  const Array<Expr> &shape2) {
    bool need_swap = false;
    Array<Expr> shape_tmp;
    Array<Expr> shape_out;
    Array<Expr> shape_long = shape1;
    Array<Expr> shape_short = shape2;
    if (shape1.size() < shape2.size()) {
      shape_long = shape2;
      shape_short = shape1;
      need_swap = true;
    }

    size_t diff_dims = shape_long.size() - shape_short.size();
    for (size_t i = 0; i < diff_dims; ++i) {
      shape_tmp.push_back(Expr(1));
      shape_out.push_back(shape_long[i]);
    }
    for (size_t i = 0; i < shape_short.size(); ++i) {
      if (GetDim(shape_long, i + diff_dims) != GetDim(shape_short, i) && !is_one(shape_long[i + diff_dims]) &&
          !is_one(shape_short[i])) {
        LOG(FATAL) << "Can not broadcast with dimension " << shape_long[i + diff_dims] << " and " << shape_short[i];
      }

      shape_tmp.push_back(shape_short[i]);

      if (is_one(shape_long[i + diff_dims])) {
        shape_out.push_back(shape_short[i]);
      } else {
        shape_out.push_back(shape_long[i + diff_dims]);
      }
    }

    if (need_swap) {
      return std::make_tuple(shape_tmp, shape_long, shape_out);
    }
    return std::make_tuple(shape_long, shape_tmp, shape_out);
  }
};

// When Matmul has DefaultFormat bias, reshape bias to FRACTAL_NZ format
// If bias need pad, do pad as
// input_2_reshape(1,1,1,16) = Reshape(input_2(2)):float16:PI
class ReshapeMatmul : public ReshapeTensorMutator {
 public:
  explicit ReshapeMatmul() = default;
  ~ReshapeMatmul() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    attr_level_++;
    Map<std::string, NodeRef> attr_map;
    GatherAttrs(op, attr_map);
    if (attr_map.count("transpose_b")) {
      transpose_b_.push(attr_map["transpose_b"].as<IntImm>()->value);
    }
    auto stmt = this->Mutate(op->body);
    if (attr_map.count("transpose_b")) {
      transpose_b_.pop();
    }
    stmt = ModifyAttrMap(op, stmt, attr_map);
    return AddReshape(stmt);
  }

  bool CheckList(const std::string &name) override {
    static std::unordered_set<std::string> check_list = {"MatMul", "BatchMatMul"};
    return check_list.find(name) != check_list.end();
  }

  bool CheckArgs(const Array<Expr> &args) override {
    // Check matmul has bias
    return args.size() < 3;
  }

  bool GetShapes(const Call *call, Array<Expr> &shape_fractal, Array<Expr> &shape_default, bool &need_swap) override {
    auto args = call->args;
    auto op_name = call->name;

    Tensor tensor1 = RecoverTensor(args[1]);
    Tensor tensor2 = RecoverTensor(args[2]);
    std::string format1;
    std::string format2;
    if (!GetFormats(op_name, tensor1, tensor2, format1, format2)) {
      return false;
    }
    if (format1 == "FRACTAL_NZ" && format2 == "DefaultFormat") {
      shape_fractal = tensor1->shape;
      shape_default = tensor2->shape;
      PadBias(shape_default);
      need_swap = false;
    } else {
      LOG(WARNING) << "[" << op_name << "], check whether input formats are valid";
      return false;
    }

    return true;
  }

  Array<Expr> InferShapeFromFractalNz(const Array<Expr> &fractal) override {
    auto orig_shape = ReshapeTensorMutator::InferShapeFromFractalNz(fractal);
    if (!transpose_b_.empty() && transpose_b_.top()) {
      Expr k = orig_shape[orig_shape.size() - 1];
      Expr n = orig_shape[orig_shape.size() - 2];
      orig_shape.Set(orig_shape.size() - 1, n);
      orig_shape.Set(orig_shape.size() - 2, k);
    }
    return orig_shape;
  }

  Array<Expr> InferShapeToFractalNz(const Array<Expr> &shape0, const Array<Expr> &shape1, const Array<Expr> &shape_out,
                                    const Array<Expr> &shape_fractal, const std::string &op_name,
                                    const Array<Expr> &shape_default) override {
    auto dims = shape_out.size();
    auto batch = dims - 2;
    Array<Expr> shape_new;
    CHECK(dims >= 2);
    for (size_t i = 0; i < batch; ++i) {
      shape_new.push_back(shape_out[i]);
    }
    if (is_one(shape1[dims - OFFSET2]) && GetDim(shape1, dims - OFFSET1) == GetDim(shape0, dims - OFFSET1)) {
      if (!transpose_b_.empty() && transpose_b_.top()) {
        shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET3]);
        shape_new.push_back(Expr(1));
        shape_new.push_back(Expr(1));
        shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET2]);
      } else {
        shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET4]);
        shape_new.push_back(Expr(1));
        shape_new.push_back(Expr(1));
        shape_new.push_back(shape_fractal[shape_fractal.size() - OFFSET1]);
      }
    } else {
      LOG(FATAL) << "[" << op_name << "] " << shape_fractal << " (FRACTAL_NZ) and " << shape_default
                 << " (DefaultFormat) may need data format transformation for ";
    }
    return shape_new;
  }

  Array<Tensor> CollectInputTensors(bool need_swap, const Array<Expr> &args, const Array<Expr> &shape_new) override {
    Array<Tensor> input_tensors;
    input_tensors.push_back(RecoverTensor(args[0]));
    input_tensors.push_back(RecoverTensor(args[1]));
    Tensor tensor2 = RecoverTensor(args[2]);
    auto reshape_tensor = placeholder(shape_new, tensor2->dtype, tensor2->op->name + "_reshape");
    reshape_.Set(tensor2, reshape_tensor);
    input_tensors.push_back(reshape_tensor);
    return input_tensors;
  }

 private:
  std::stack<bool> transpose_b_;

  void PadBias(Array<Expr> &shape_default) {
    if (shape_default.size() != 1) {
      return;
    }
    auto bias_length = (shape_default[0].as<IntImm>())->value;
    if (bias_length % 16 == 0) {
      return;
    }
    int64_t pad_length = (bias_length / 16) * 16 + 16;
    shape_default.Set(0, Expr(pad_length));
    LOG(INFO) << "Pad bias length from " << bias_length << " to " << pad_length;
  }
};

Stmt ReshapeTensor(const Stmt &s, BuildInfo *) {
  auto stmt = ReshapeTensorMutator().Mutate(s);
  return ReshapeMatmul().Mutate(stmt);
}
}  // namespace akg
