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
class InplaceAssignMutator : public IRMutator {
 public:
  explicit InplaceAssignMutator(BuildOpt &opt) : opt_(opt) {}

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs") {
      op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
      auto stmt = IRMutator::Mutate_(op, s);
      op_attrs_ = {};
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    auto op_name = call->name;
    if (op_name == "InplaceAssign") {
      if (op_attrs_.count("fake_output")) {
        auto fake_val = op_attrs_["fake_output"].as<IntImm>();
        if (fake_val && fake_val->value > 0) {
          opt_.fakeout.insert(op->func);
        }
      }
      auto inputs = call->args;
      opt_.sames[op->func] = inputs[2].as<Call>()->func;  // d = InplaceAssign(a, b, c)     d = c
      if (auto i1 = inputs[1].as<Call>()) {
        opt_.inplaces[i1->func] = inputs[0];  // d = InplaceAssign(a, b, c)     a = b
        return Evaluate::make(0);
      } else {
        // d = Assign(dst, src)    d = dst   fake d, d should be InplaceAssigin's inputs[2]
        return Provide::make(op->func, op->value_index,
                             Call::make(call->type, "Assign", {inputs[0], inputs[1]}, call->call_type), op->args);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  BuildOpt &opt_;
  Map<std::string, NodeRef> op_attrs_;
};

class AssignToInplaceAssignMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    Map<std::string, NodeRef> attrs;
    if (op->attr_key == "attrs") {
      attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      is_assign_ = false;
      has_attrs_ = true;
      auto body = this->Mutate(op->body);
      has_attrs_ = false;
      if (!is_assign_) {
        return s;
      }
      is_assign_ = false;
      attrs.Set("fake_output", make_const(Int(1), true));
      return AttrStmt::make(attrs, "attrs", Expr(1), body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    auto op_name = call->name;
    if (op_name == "Assign") {
      is_assign_ = true;
      auto &inputs = call->args;
      auto p = Provide::make(
        op->func, op->value_index,
        Call::make(call->type, "InplaceAssign", {inputs[0], inputs[1], inputs[1]}, call->call_type), op->args);
      if (has_attrs_) {
        return p;
      }
      Map<std::string, NodeRef> attrs;
      attrs.Set("fake_output", make_const(Int(1), true));
      return AttrStmt::make(attrs, "attrs", Expr(1), p);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool has_attrs_{false};
  bool is_assign_{false};
};

Stmt AssignToInplaceAssign(const Stmt &s, BuildInfo *) { return AssignToInplaceAssignMutator().Mutate(s); }
Stmt InplaceAssignOpt(const Stmt &s, BuildInfo *info) { return InplaceAssignMutator(info->opt).Mutate(s); }
}  // namespace akg
