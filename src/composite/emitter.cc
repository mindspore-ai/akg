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
#include "composite/emitter.h"

namespace akg {
void Emitter::Visit_(const AttrStmt *op) {
  if (op->attr_key == "attrs") {
    op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
    Visit(op->body);
    op_attrs_ = {};
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
      array_result_[op->func] = (*topi_f)(real_inputs, op_attrs_);
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
        EmitAssign(t, Downcast<Expr>(real_inputs[0]));
      }
      CollectNoinlineCandidate(real_inputs, t);
      opt_.tensor_map[op->func] = t;
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

void Emitter::EmitAssign(Tensor &t, const Expr &input) {
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
