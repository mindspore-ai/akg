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

#include "composite/optimize/pass.h"

namespace akg {

class AddAttrsForOpMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    auto new_body = Mutate(op->body);
    CHECK_EQ(op->attr_key, "attrs");
    auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    if (op_name_ == "Reshape") {  // reshape's attr may have shape [-1], it will cause error.
      attrs.Set("shape", provide_->args);
    }
    if (op_name_ == "OneHot") {
      CHECK(provide_->value.as<Call>());
      auto call = provide_->value.as<Call>();
      attrs.Set("dst_type", Expr(type2string(call->type)));
    }
    return AttrStmt::make(attrs, op->attr_key, op->value, new_body);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) {
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    op_name_ = call->name;
    provide_ = op;
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::string op_name_;
  const Provide *provide_{nullptr};
};

Stmt AddAttrsForOp(const Stmt &s, BuildInfo *) { return AddAttrsForOpMutator().Mutate(s); }
}  // namespace akg
