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
bool CantInline(const Operation &op) {
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

void AutoInline(Schedule sch) {
  // Note: do not support inline of hybrid ops
  std::unordered_set<Operation, NodeHash, NodeEqual> uninlinable;
  for (const Stage &s : sch->stages) {
    if (const auto op = s->op.as<air::HybridOpNode>()) {
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

  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && (IsInjective(s->op) || air::schedule::IsElemWise(s->op)) && !CantInline(s->op) &&
        !s->is_output && uninlinable.count(s->op) == 0 && !(has_conv && !IsConvInline(s->op, conv_inputs)) &&
        (s->op->attrs.count("no_inline") == 0)) {
      static_cast<void>(s.compute_inline());
    }
  }
}
}  // namespace schedule
}  // namespace akg
