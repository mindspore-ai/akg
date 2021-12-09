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
/*
 * 2021.10.28 - Add auto inline logic for hybrid and extern op
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
      if (op.second > 1) {
        RemoveShortCommonExpr(op.first, op.first);
      }
    }
    std::unordered_set<Operation> common_op;
    for (const auto op : counter) {
      if (op.first.as<ComputeOpNode>() != nullptr && op.second > 1) {
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

  void RemoveShortCommonExpr(const Operation op, const Operation root) {
    if (const auto cur_op = op.as<ComputeOpNode>()) {
      for (const auto parent : cur_op->InputTensors()) {
        if (counter.count(parent->op) && counter[root] > 3) {
          counter[parent->op] -= counter[root];
        }
        RemoveShortCommonExpr(parent->op, root);
      }
    }
  }

  std::unordered_set<Operation> op_list;
  std::unordered_map<Operation, int> counter;
};

std::unordered_set<Operation, NodeHash, NodeEqual> GetFilterOp(const Array<Expr> &filter_exprs) {
  std::unordered_set<Operation, NodeHash, NodeEqual> filter_op;
  for (const auto &arg : filter_exprs) {
    PostOrderVisit(arg, [&filter_op](const NodeRef &node) {
      if (auto call = node.as<Call>()) {
        if (call->func.defined() && call->func->IsInstance<ComputeOpNode>()) {
          filter_op.insert(Downcast<Operation>(call->func));
        }
      }
    });
  }
  return filter_op;
}

class GetCubeMatmulInput : public IRVisitor {
 public:
  Array<Expr> Run(Schedule sch) {
    for (Stage s : sch->stages) {
      auto op = s->op;
      if (op->tag == "dense" || op->tag == "batch_matmul" || op->tag == "matmul") {
        auto compute_op = op.as<ComputeOpNode>();
        CHECK(compute_op);
        for (const auto &e : compute_op->body) {
          Visit(e);
        }
      }
    }
    return input_exprs_;
  }

 private:
  void Visit_(const Reduce *op) override {
    if (op->combiner.defined() && op->combiner->result.size() > 0) {
      auto call = op->combiner->result[0].as<Call>();
      // "mad" : call of cube matmul in the cce
      if (call && call->name == "mad" && call->call_type == Call::PureIntrinsic) {
        auto mad_real_args = op->source;
        for (auto arg : mad_real_args) {
          input_exprs_.push_back(arg);
        }
      }
    }
    return IRVisitor::Visit_(op);
  }

  Array<Expr> input_exprs_;
};

void AutoInline(Schedule sch, const Target &target, bool enable_cse) {
  std::unordered_set<Operation, NodeHash, NodeEqual> uninlinable;
  std::unordered_set<Operation, NodeHash, NodeEqual> no_inject_inline;
  for (const Stage &s : sch->stages) {
    if (const auto op = s->op.as<air::HybridOpNode>()) {
      // disable inline the inputs of an op in the following two cases:
      // 1. if the op has the attr disable_inline_inject,
      //    that is the op refusing any inline injecting from inputs
      // 2. if the target is cce, as any inline inputs will be recreated
      //    by the tothreeaddress pass
      if (op->attrs.count("disable_inline_inject")) {
        for (Tensor t : op->inputs) {
          if (!t->op->IsInstance<PlaceholderOpNode>()) no_inject_inline.insert(t->op);
        }
      } else if (target->device_type == kDLCce) {
        for (Tensor t : op->inputs) {
          uninlinable.insert(t->op);
        }
      }
    }
    if (const auto op = s->op.as<air::ExternOpNode>()) {
      if (op->attrs.count("disable_inline_inject")) {
        for (Tensor t : op->inputs) {
          if (!t->op->IsInstance<PlaceholderOpNode>()) no_inject_inline.insert(t->op);
        }
      } else if (target->device_type == kDLCce) {
        for (Tensor t : op->inputs) {
          uninlinable.insert(t->op);
        }
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
  if (target->device_type == kDLGPU && enable_cse) {
    common_subexpr = CSE().FindCommonSubexpr(sch);
  }

  // For the CCE, do not perform the inline for input exprs of some calls
  if (target->device_type == kDLCce) {
    auto input_exprs = GetCubeMatmulInput().Run(sch);
    auto filter_op = GetFilterOp(input_exprs);
    uninlinable.insert(filter_op.begin(), filter_op.end());
  }

  for (Stage s : sch->stages) {
    if (no_inject_inline.count(s->op)) {
      s->no_inline_inject = true;
      continue;
    }
    if (!s.is_scheduled() && (IsInjective(s->op) || air::schedule::IsElemWise(s->op)) && !CantInline(s->op, target) &&
        !s->is_output && uninlinable.count(s->op) == 0 && !(has_conv && !IsConvInline(s->op, conv_inputs)) &&
        (s->op->attrs.count("no_inline") == 0 && common_subexpr.count(s->op) == 0)) {
      static_cast<void>(s.compute_inline());
    }
  }
}
}  // namespace schedule
}  // namespace akg
