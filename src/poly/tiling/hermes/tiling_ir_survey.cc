/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <utility>

#include "poly/tiling/hermes/tiling_ir_survey.h"

namespace akg {
namespace ir {
namespace poly {
void TilingIRSurvey::Visit_(const AttrStmt *op) {
  if (is_symbolic_enabled_) {
    if (const auto *const comp_op = op->node.as<ComputeOpNode>()) {
      for (auto attr : comp_op->attrs) {
        if (attr.first == "dim" && attr.second.as<StringImm>() != nullptr) {
          LOG(DEBUG) << "Symbolic tiling disabled: IR defined dimensions";
          is_symbolic_enabled_ = false;
          return;
        }
      }
    }
    IRVisitor::Visit_(op);
  }
}

void TilingIRSurvey::Visit_(const For *op) {
  if (is_symbolic_enabled_) {
    auto iter = for_range_.find(op->loop_var->name_hint);
    if (iter == for_range_.end()) {
      for_range_.insert(std::make_pair(op->loop_var->name_hint, op->extent.as<IntImm>()->value));
    }
    IRVisitor::Visit_(op);
    for_range_.erase(op->loop_var->name_hint);
  }
}

void TilingIRSurvey::Visit_(const Call *op) {
  if (is_symbolic_enabled_) {
    if (op->func) {
      air::Tensor tensor = Downcast<Operation>(op->func).output(op->value_index);
      int idx = 0;
      for (auto const &arg : op->args) {
        if (arg.as<Variable>() == nullptr && arg.as<IntImm>() == nullptr) {
          LOG(DEBUG) << "Symbolic tiling disabled: IR call args";
          is_symbolic_enabled_ = false;
          return;
        }
        if (arg.as<Variable>() != nullptr &&
            for_range_.find(arg.as<Variable>()->name_hint)->second != tensor->shape[idx].as<IntImm>()->value) {
          LOG(DEBUG) << "Symbolic tiling disabled: IR for extent different than call tensor shape";
          is_symbolic_enabled_ = false;
          return;
        }
        ++idx;
      }
    }
    IRVisitor::Visit_(op);
  }
}

void TilingIRSurvey::Visit_(const Provide *op) {
  if (is_symbolic_enabled_) {
    air::Tensor tensor = Downcast<Operation>(op->func).output(op->value_index);
    int idx = 0;
    for (auto const &arg : op->args) {
      if (arg.as<Variable>() != nullptr) {
        if (for_range_.find(arg.as<Variable>()->name_hint)->second != tensor->shape[idx].as<IntImm>()->value) {
          LOG(DEBUG) << "Symbolic tiling disabled: IR for extent different than provide tensor shape";
          is_symbolic_enabled_ = false;
          return;
        }
        provide_args_.push_back(arg.as<Variable>()->name_hint);
      }
      ++idx;
    }
    IRVisitor::Visit_(op);
    provide_args_.clear();
  }
}

void TilingIRSurvey::Visit_(const Realize *op) {
  if (is_symbolic_enabled_) {
    std::stringstream func_stream;
    func_stream << op->func;
    if (func_stream.str().rfind("hybrid", 0) == 0) {
      LOG(DEBUG) << "Symbolic tiling disabled: hybrid IR";
      is_symbolic_enabled_ = false;
      return;
    }
    IRVisitor::Visit_(op);
  }
}

bool HasSymbolicStatusChanged(const Stmt &stmt_sch) {
  auto symbolic_survey = TilingIRSurvey();
  symbolic_survey.Visit(stmt_sch);
  return !symbolic_survey.IsSymbolicEnabled();
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
