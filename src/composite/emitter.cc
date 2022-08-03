/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <vector>
#include <regex>
#include "composite/emitter.h"

namespace akg {
constexpr auto kEnableAutoInplace = "enable_auto_inplace";

// particular used for custom op
// arg s is in the form of "0 1 1 -1 2 -1" and should be translated into {{0,1},{1,-1},{2,-1}}
// {0,1} means outputs[0] and inputs[1] are in inplace relation; {1,-1} means outputs[1] has no inplace relation
// tensors in inplace relation should be binded together and share the same buffer
std::vector<std::vector<int>> parse_inplace_str(const std::string &s) {
  std::regex delimiters(" ");
  std::vector<std::string> index(std::sregex_token_iterator(s.begin(), s.end(), delimiters, -1),
                                 std::sregex_token_iterator());
  std::vector<std::vector<int>> inplace_index;
  std::vector<int> tmp;
  for (size_t i = 0; i < index.size(); i++) {
    tmp.push_back(std::stoi(index[i]));
    if (i & 1) {
      if (tmp.back() >= 0) {
        inplace_index.push_back(tmp);
      }
      tmp.clear();
    }
  }
  return inplace_index;
}

void Emitter::Visit_(const AttrStmt *op) {
  if (op->attr_key == "attrs") {
    op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
    // attr inplace_assign_output is from info
    if (op_attrs_.find("inplace_assign_output") != op_attrs_.end()) {
      // an example of s: "0 1 1 -1 2 -1"
      if (auto s = op_attrs_["inplace_assign_output"].as<StringImm>()) {
        inplace_relation_ = parse_inplace_str(s->value);
      }
    }
    Visit(op->body);
    op_attrs_ = {};
    inplace_relation_.clear();
  } else {
    IRVisitor::Visit_(op);
  }
}

void Emitter::Visit_(const Provide *op) {
  CHECK(op->value.as<Call>());
  auto call = op->value.as<Call>();
  op_name_ = call->name;
  if (op_name_ == "tuple_getitem") {
    ProcessTupleGetItem(op, call->args);
    return;
  }
  EmitTopi(op, GetRealInputs(call->args));
}

void Emitter::ProcessTupleGetItem(const Provide *op, const Array<Expr> &inputs) {
  // tuple_getitem is called when getting items from multi result array
  CHECK(inputs[0].as<Call>());
  // the first input of tuple_getitem is the placeholder node of the array result
  // get the real array result from array_result_ map
  auto result_tuple_call = inputs[0].as<Call>()->func;
  CHECK(array_result_.count(result_tuple_call));
  Array<Tensor> tuple_result = array_result_[result_tuple_call];
  // the second input of tuple_getitem is the index
  // get the i'th result from the array result
  NodeRef index = inputs[1];
  CHECK(index->IsInstance<UIntImm>());
  auto value_index = index.as<UIntImm>()->value;
  Tensor t = tuple_result[value_index];
  opt_.tensor_map[op->func] = t;
  if (array_inplace_.find(static_cast<int>(value_index)) != array_inplace_.end()) {
    opt_.inplaces[op->func] = array_inplace_[static_cast<int>(value_index)];
    opt_.fakeout.insert(op->func);
  }
}

Array<NodeRef> Emitter::GetRealInputs(const Array<Expr> &inputs) {
  Array<NodeRef> real_inputs;
  for (const auto &input : inputs) {
    if (auto c = input.as<Call>()) {
      if (opt_.tensor_map.count(c->func) == 0) {
        Tensor t = placeholder(c->args, c->type, c->name);
        opt_.tensor_map[c->func] = t;
      }
      real_inputs.push_back(opt_.tensor_map[c->func]);
    } else {
      real_inputs.push_back(input);
    }
  }
  return real_inputs;
}

void Emitter::EmitTopi(const Provide *op, const Array<NodeRef> &real_inputs) {
  const auto *topi_f = GetTopiFunc();
  if (auto placeholder = op->func.as<air::PlaceholderOpNode>()) {
    if (placeholder->dtype.code() == kArrayHandle) {
      // in this case, the output is an array of tensor
      // store the result in array_result_ map
      Array<Tensor> res = (*topi_f)(real_inputs, op_attrs_);
      array_result_[op->func] = res;
      array_inplace_.clear();
      if (!inplace_relation_.empty()) {
        for (auto &index : inplace_relation_) {
          auto input_index = index[1];
          auto output_index = index[0];
          CHECK_LT(input_index, real_inputs.size())
            << "Given input index: " << input_index << " with total " << real_inputs.size() << " inputs";
          CHECK_LT(output_index, res.size())
            << "Given output index: " << output_index << " with total " << res.size() << " outputs";
          array_inplace_[output_index] = real_inputs[input_index];
        }
      }

      return;
    } else {
      NodeRef res = (*topi_f)(real_inputs, op_attrs_);
      Tensor t;
      if (res->IsInstance<TensorNode>()) {
        t = Downcast<Tensor>(res);
      } else {
        auto val = Downcast<Expr>(res);
        auto fcompute = [&val](const Array<Var> &indices) { return val; };
        t = compute(Array<Expr>{1}, fcompute, "broadcast");
      }
      if (op_name_ == "Assign") {
        EmitAssign(t, real_inputs[0]);
      }
      if (inplace_relation_.size() == 1) {
        auto input_index = inplace_relation_[0][1];
        CHECK_LT(input_index, real_inputs.size())
          << "Given input index: " << input_index << " with total " << real_inputs.size() << " inputs";
        opt_.inplaces[op->func] = real_inputs[input_index];
        opt_.fakeout.insert(op->func);
      }
      CollectNoinlineCandidate(real_inputs, t);
      opt_.tensor_map[op->func] = t;
      if (op_attrs_.count(kEnableAutoInplace) > 0) {
        opt_.tensor_attrs[t].Set(kEnableAutoInplace, op_attrs_[kEnableAutoInplace]);
      }
    }
  } else {
    LOG(FATAL) << "Unexpected op func type: " << op->func;
  }
}

const PackedFunc *Emitter::GetTopiFunc() {
  const auto *topi_f = air::runtime::Registry::Get(op_name_);
  if (topi_f == nullptr && !opt_.target.empty()) {
    std::string target = opt_.target;
    target[0] = std::toupper(target[0]);
    topi_f = air::runtime::Registry::Get(target + op_name_);
  }
  CHECK(topi_f) << "Akg topi has no op: " << op_name_;
  return topi_f;
}

void Emitter::EmitAssign(Tensor &t, const NodeRef &input) {
  // copy out to bind_input, bind_input is used to bind input[0]
  // d = Assign(a, b), bind_input = d, input0 = bind_input
  auto bind_input = compute(
    t->shape, [&](const Array<Var> &indices) { return t(indices); }, "assign_tensor_" + std::to_string(assign_count_));
  opt_.tensor_map[bind_input->op] = bind_input;
  opt_.sch_only.emplace_back(bind_input);
  opt_.inplaces[bind_input->op] = input;
  assign_count_++;
}

void Emitter::CollectNoinlineCandidate(const Array<NodeRef> &real_inputs, const Tensor &t) {
#ifdef ENABLE_GENERAL_TOT
  if (t->op->tag != "tot") {
    return;
  }
  const auto &attrs = t->op->attrs;
  auto no_inline = ir::GetInt32Const(Downcast<Expr>(attrs["no_inline"]));
  NodeRef arg(t);
  if (no_inline != -1) {
    arg = real_inputs[no_inline];
  }
  auto iter = std::find_if(opt_.noinline_candidate.begin(), opt_.noinline_candidate.end(),
                           [&arg](const Tensor &cand) { return Downcast<Tensor>(arg)->op->name == cand->op->name; });
  if (iter == opt_.noinline_candidate.end()) {
    opt_.noinline_candidate.push_back(Downcast<Tensor>(arg));
  }
#endif
  return;
}

}  // namespace akg
