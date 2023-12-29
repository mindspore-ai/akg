/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "tvm.h"
#include <pass/storage_access.h>

namespace akg {
namespace ir {
class BypassActor : public IRMutator {
 public:
  BypassActor() {}
  ~BypassActor() override = default;

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (!filter_func.empty() && (filter_func.front() == op->func)) {
      const auto r = stmt.as<Realize>();
      CHECK(r != nullptr);
      stmt = r->body;
      filter_func.clear();
    }

    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_bypass_filter_l1") {
      // clear offsets and set compute offset flag
      offsets.clear();
      bypass_filter_l1 = true;
      static_cast<void>(IRMutator::Mutate(op->body));
      bypass_filter_l1 = false;

      const auto produce = op->body.as<ProducerConsumer>();
      CHECK(produce != nullptr);
      filter_func.push_back(produce->func);

      return Evaluate::make(0);
    }

    if (op->attr_key == "pragma_bypass_filter_l0") {
      // set replace flag
      doing_replace = true;
      auto stmt = IRMutator::Mutate(op->body);
      doing_replace = false;

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    // compute offset
    if (bypass_filter_l1) {
      const Call *call = op->value.as<Call>();
      CHECK(call != nullptr);
      callName = call->name;
      func = call->func;

      air::arith::Analyzer analyzer_;
      for (size_t i = 0; i < call->args.size(); ++i) {
        offsets.push_back(analyzer_.Simplify(call->args[i] - op->args[i]));
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    // replace offset
    if (doing_replace) {
      Array<Expr> args;
      air::arith::Analyzer analyzer_;
      for (size_t i = 0; i < op->args.size(); ++i) {
        args.push_back(analyzer_.Simplify(offsets[i] + op->args[i]));
      }

      auto ee = Call::make(op->type, callName, args, Call::CallType::Halide, func, op->value_index);

      return IRMutator::Mutate_(op, ee);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  bool bypass_filter_l1{false};
  bool doing_replace{false};

  std::vector<Expr> offsets;
  std::string callName;
  FunctionRef func;
  std::vector<FunctionRef> filter_func;
};

Stmt BypassL1(const Stmt &stmt) { return BypassActor().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
