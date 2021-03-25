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
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <tvm.h>
#include <vector>
#include <stack>

namespace air {
namespace schedule {
bool IsElemWise(const Operation &op);
}  // namespace schedule
}  // namespace air

namespace akg {
namespace schedule {
using air::Stage;

// filter for op can not be inlined
class InlineFilter : public IRVisitor {
 public:
  InlineFilter() {}
  ~InlineFilter() override = default;

  void Visit_(const Call *op) final {
    if (op->name == "tvm_if_then_else" || op->name == "proposal_sort") {
      is_filtered_ = true;
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Cast *op) final {
    if (op->value.type() == Int(8) || op->value.type() == UInt(8)) {
      is_filtered_ = true;
      return;
    }
  }

  void Visit_(const Select *op) final {
    is_filtered_ = true;
    return;
  }

  bool is_filtered_{false};
};

// op can not be inlined
bool CantInline(const Operation &op, const Target &target) {
  if (target->device_type == kDLGPU) {
    return false;
  }
  if (const auto compute = op.as<ComputeOpNode>()) {
    InlineFilter v;
    for (auto &e : compute->body) {
      v.Visit(e);
      if (v.is_filtered_) {
        return true;
      }
    }
  }
  return false;
}

bool IsInjective(const Operation &op) {
  if (const auto compute = op.as<ComputeOpNode>()) {
    return compute->reduce_axis.size() == 0;
  }
  return false;
}

bool IsConvInline(const Operation &op, const std::unordered_set<std::string> &inputs) {
  if (const auto compute = op.as<ComputeOpNode>()) {
    if (inputs.count(compute->name)) {
      bool can_inline = true;
      for (auto &e : compute->body) {
        auto call = e.as<Call>();
        if (!(call && call->call_type == Call::Halide)) {
          can_inline = false;
        }
      }
      return can_inline;
    }
  }
  return false;
}

class ConvInputDetector : public IRVisitor {
 public:
  ConvInputDetector() {}
  ~ConvInputDetector() override = default;

  void Visit_(const Call *op) final {
    if (op->call_type == Call::Halide) {
      conv_inputs_.insert(op->name);
      return;
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_set<std::string> conv_inputs_;
};

void GetConvInputName(const Operation &op, std::unordered_set<std::string> &inputs) {
  if (const auto compute = op.as<ComputeOpNode>()) {
    ConvInputDetector v = ConvInputDetector();
    for (auto &e : compute->body) {
      v.Visit(e);
    }
    inputs = v.conv_inputs_;
  }
}

class CSE {
 public:
  std::unordered_set<Operation> FindCommonSubexpr(Schedule sch) {
    for (const Stage &s : sch->stages) {
      op_list.insert(s->op);
    }
    for (const auto op : sch->outputs) {
      count_used_number(op);
    }
    for (const auto op : counter) {
      if (op.second > 1){
        RemoveShortCommonExpr(op.first, op.first);
      }
    }
    std::unordered_set<Operation> common_op;
    for (const auto op : counter) {
      int input_num = 0;
      input_num = count_input_number(op.first, input_num);
      if (op.first.as<ComputeOpNode>() != nullptr && op.second > 1 && input_num > 2) {
        common_op.insert(op.first);
      }
    }
    return common_op;
  }

 private:
  void count_used_number(const Operation &op) {
    if (const auto compute = op.as<ComputeOpNode>()) {
      for (const auto parent : op->InputTensors()) {
        if (op_list.count(parent->op) == 0) continue;
        if (counter.count(parent->op) == 0) counter[parent->op] = 0;
        counter[parent->op]++;
        count_used_number(parent->op);
      }
    }
  }

  int count_input_number(const Operation op, int input_num) {
    if (op.as<PlaceholderOpNode>() != nullptr) return (input_num + 1);
    if (const auto compute = op.as<ComputeOpNode>()) {
      for (const auto parent : op->InputTensors()) {
        input_num = count_input_number(parent->op, input_num);
      }
    }
    return input_num;
  }

  void RemoveShortCommonExpr(const Operation op, const Operation root) {
    if (const auto cur_op  = op.as<ComputeOpNode>()) {
      for (const auto parent : cur_op->InputTensors()) {
        if (counter.count(parent->op) && counter[parent->op] > 0) {
          counter[parent->op] -= counter[root];
        }
        RemoveShortCommonExpr(parent->op, root);
      }
    }
  }

  std::unordered_set<Operation> op_list;
  std::unordered_map<Operation, int> counter;
};

void AutoInline(Schedule sch, const Target &target ,bool enable_cse) {
  // Note: do not support inline of hybrid ops and extern ops
  std::unordered_set<Operation, NodeHash, NodeEqual> uninlinable;
  for (const Stage &s : sch->stages) {
    if (const auto op = s->op.as<air::HybridOpNode>()) {
      for (Tensor t : op->inputs) {
        uninlinable.insert(t->op);
      }
    }
    if (const auto op = s->op.as<air::ExternOpNode>()) {
      for (Tensor t : op->inputs) {
        uninlinable.insert(t->op);
      }
    }
  }

  bool has_conv = false;
  std::unordered_set<std::string> conv_inputs;
  for (const Stage &s : sch->stages) {
    if ((s->op->attrs.count("feature")) && (s->op->attrs.count("filter"))) {
      has_conv = true;
      GetConvInputName(s->op, conv_inputs);
    }
  }

  std::unordered_set<Operation> common_subexpr;
  if (target->device_type == kDLGPU && enable_cse){
    common_subexpr = CSE().FindCommonSubexpr(sch);
  }

  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && (IsInjective(s->op) || air::schedule::IsElemWise(s->op)) && !CantInline(s->op, target) &&
        !s->is_output && uninlinable.count(s->op) == 0 && !(has_conv && !IsConvInline(s->op, conv_inputs)) &&
        (s->op->attrs.count("no_inline") == 0 && common_subexpr.count(s->op) == 0)) {
      static_cast<void>(s.compute_inline());
    }
  }
}
}  // namespace schedule
}  // namespace akg
