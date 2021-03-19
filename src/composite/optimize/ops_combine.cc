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

#include "composite/optimize/ops_combine.h"
#include <string>
#include <unordered_map>
#include <utility>
#include "dmlc/common.h"

namespace akg {
namespace {
using CmpInfo = std::pair<std::string, Array<Expr>>;
using CmpFuncMap = std::unordered_map<std::string, std::string>;
using CondArgsMap = std::unordered_map<FunctionRef, CmpInfo, NodeHash, NodeEqual>;
using RefCntMap = std::unordered_map<FunctionRef, int, NodeHash, NodeEqual>;
class CmpSelVisitor : public IRVisitor {
 public:
  std::pair<CondArgsMap, RefCntMap> Process(const Stmt &s) {
    IRVisitor::Visit(s);
    return make_pair(condition_with_inputs_, output_ref_cnt_);
  }

 private:
  void Visit_(const Call *op) final {
    if (output_ref_cnt_.find(op->func) != output_ref_cnt_.end()) {
      ++output_ref_cnt_[op->func];
    }
    return IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) final {
    auto call = op->value.as<Call>();
    if (call == nullptr) {
      return IRVisitor::Visit_(op);
    }

    if (cmp_funcs_.find(call->name) != cmp_funcs_.end()) {
      condition_with_inputs_.insert(std::make_pair(op->func, std::make_pair(cmp_funcs_[call->name], call->args)));
      output_ref_cnt_.insert(std::make_pair(op->func, 0));
    } else if (call->name == "Select") {
      auto cond_call = call->args[0].as<Call>();
      if (cond_call && condition_with_inputs_.find(cond_call->func) != condition_with_inputs_.end()) {
        --output_ref_cnt_[cond_call->func];
      }
    }
    return IRVisitor::Visit_(op);
  }

  static CmpFuncMap cmp_funcs_;
  CondArgsMap condition_with_inputs_;
  RefCntMap output_ref_cnt_;
};

std::unordered_map<std::string, std::string> CmpSelVisitor::cmp_funcs_ = {
  {"GreaterEqual", "GE"}, {"Greater", "GT"}, {"LessEqual", "LE"}, {"Less", "LT"}, {"Equal", "EQ"}, {"NotEqual", "NE"}};

class CmpSelMutator : public IRMutator {
 public:
  CmpSelMutator(const CondArgsMap &condition_with_inputs, const RefCntMap output_ref_cnt,
                const std::vector<std::string> &output_tensors)
      : condition_with_inputs_(condition_with_inputs),
        output_ref_cnt_(output_ref_cnt),
        output_tensors_(output_tensors) {}

 private:
  bool IsIsolated(const FunctionRef &func) {
    auto iter = output_ref_cnt_.find(func);
    return iter != output_ref_cnt_.end() && iter->second < 1 &&
           std::find(output_tensors_.begin(), output_tensors_.end(), func->func_name()) == output_tensors_.end();
  }

  Array<Expr> GetFusionOpInputs(const Array<Expr> &inputs) {
    Array<Expr> fusion_inputs;
    for (const auto &item : inputs) {
      if (auto c = item.as<Call>()) {
        if (condition_with_inputs_.find(c->func) != condition_with_inputs_.end()) {
          for (auto input : condition_with_inputs_.find(c->func)->second.second) {
            fusion_inputs.push_back(input);
          }
          continue;
        }
      }
      fusion_inputs.push_back(item);
    }
    return fusion_inputs;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (IsIsolated(op->func)) {
      return Evaluate::make(0);
    }

    auto call = op->value.as<Call>();
    if (call && call->name == "Select") {
      auto cond_call = call->args[0].as<Call>();
      if (cond_call && condition_with_inputs_.find(cond_call->func) != condition_with_inputs_.end()) {
        auto fusion_inputs = GetFusionOpInputs(call->args);
        auto combine_op_name = "Select" + condition_with_inputs_.find(cond_call->func)->second.first;
        return Provide::make(op->func, op->value_index,
                             Call::make(Int(32), combine_op_name, fusion_inputs, Call::CallType::PureIntrinsic),
                             op->args);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  const CondArgsMap &condition_with_inputs_;
  const RefCntMap &output_ref_cnt_;
  const std::vector<std::string> &output_tensors_;
};
}  // namespace

/*
 * Combine Select and its condition. e.g.
 *
 * output_0(1) = Greater(input_0(1), input_1(1))
 * output(1) = Select(output_0(1), input_2(1), input_3(1))
 * ------------>
 * output(1) = SelectGT(input_0(1), input_1(1), input_2(1), input_3(1))
 */
class SelectFusion {
 public:
  explicit SelectFusion(const std::vector<std::string> &output_tensors) : output_tensors_(output_tensors) {}
  Stmt DoFusion(const Stmt &s) {
    CondArgsMap cond_args;
    RefCntMap ref_cnt;
    std::tie(cond_args, ref_cnt) = CmpSelVisitor().Process(s);
    return CmpSelMutator(cond_args, ref_cnt, output_tensors_).Mutate(s);
  }

 private:
  const std::vector<std::string> &output_tensors_;
};

Stmt OpsCombine::Run(const Stmt &s) { return SelectFusion(info_.output_names).DoFusion(s); }
}  // namespace akg
