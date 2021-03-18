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

#include "composite/optimize/select_fusion.h"
#include "dmlc/common.h"

namespace akg {
class FusionMutator : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "fusion" && op->value.as<StringImm>()) {
      fusion_op_name_ = op->value.as<StringImm>()->value;
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    if (!fusion_op_name_.empty()) {
      CHECK(op->value.as<Call>());
      auto call = op->value.as<Call>();
      if (fusion_op_name_.find("_end") == std::string::npos) {
        if (call->name == "ZerosLike") {  // ZerosLike directly transform to zero
          CHECK_EQ(call->args.size(), 1);
          CHECK(call->args[0].as<Call>());
          output_with_inputs_[op->func] = {make_zero(call->args[0].as<Call>()->type)};
        } else {
          output_with_inputs_[op->func] = call->args;
        }
        return Evaluate::make(0);
      } else {  // fusion end
        Array<Expr> fusion_inputs;
        GetFusionOpInputs(call->args, fusion_inputs);
        auto str_list = dmlc::Split(fusion_op_name_, '_');
        CHECK(!str_list.empty());
        auto stmt =
          Provide::make(op->func, op->value_index,
                        Call::make(Int(32), str_list[0], fusion_inputs, Call::CallType::PureIntrinsic), op->args);
        fusion_op_name_.clear();
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  void GetFusionOpInputs(const Array<Expr> &inputs, Array<Expr> &fusion_inputs) {
    for (const auto &item : inputs) {
      if (auto c = item.as<Call>()) {
        if (output_with_inputs_.count(c->func) != 0) {
          for (auto input : output_with_inputs_[c->func]) {
            fusion_inputs.push_back(input);
          }
          continue;
        }
      }
      fusion_inputs.push_back(item);
    }
  }

  std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual> output_with_inputs_;
  std::string fusion_op_name_;
};

Stmt SelectFusion::Run(const Stmt &s) { return FusionMutator().Mutate(s); }
}  // namespace akg
