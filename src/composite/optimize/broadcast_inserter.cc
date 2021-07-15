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

#include "composite/optimize/broadcast_inserter.h"

namespace akg {
class BroadcastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const auto *provide = op->body.as<Provide>();
      CHECK(provide);
      auto call = provide->value.as<Call>();
      CHECK(call);
      //for unary input op
      if (call->args.size() == 1) {
        Expr arg = call->args[0];
        if (arg.as<IntImm>() || arg.as<UIntImm>() || arg.as<FloatImm>()) {
          return DoInsert(arg, 0, provide, call, op);
        }
      }
      //for op with multiple inputs
      auto it = broadcast_ops_.find(call->name);
      if (it != broadcast_ops_.end()) {
        for (size_t i = 0; i < call->args.size(); ++i) {
          if (!(it->second & (1u << i))) {
            continue;
          }
          Expr e = call->args[i];
          if (e.as<IntImm>() || e.as<UIntImm>() || e.as<FloatImm>()) {
            return DoInsert(e, i, provide, call, op);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Stmt DoInsert(const Expr &e, const size_t i, const Provide *provide, const Call *call, const AttrStmt *op) {
    Stmt first, second;
    std::string name = "broadcast_" + std::to_string(name_idx_++);
    Tensor t = placeholder(provide->args, call->type, name);
    first = Provide::make(t->op, 0, Call::make(Int(32), "BroadcastTo", {e}, Call::CallType::PureIntrinsic), t->shape);
    Map<std::string, NodeRef> attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    attrs.Set("shape", t->shape);
    first = AttrStmt::make(attrs, "attrs", Expr(1), first);
    auto args = call->args;
    args.Set(i, Call::make(t->dtype, t->op->name, t->shape, Call::CallType::Halide, t->op));
    second = Provide::make(provide->func, provide->value_index,
                           Call::make(call->type, call->name, args, call->call_type), provide->args);
    second = AttrStmt::make(op->node, op->attr_key, op->value, second);
    return Block::make(first, second);
  }

  int name_idx_ = 0;
  std::unordered_map<std::string, unsigned> broadcast_ops_ = {{"Equal", -1}, {"Select", -1}};
};

Stmt BroadcastInserter::Run(const Stmt &s) { return BroadcastInserterMutator().Mutate(s); }
}  // namespace akg
