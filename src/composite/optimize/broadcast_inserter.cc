/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
using BroadcastSet = std::unordered_set<NodeRef, air::ExprHash, air::NodeEqual>;
using SubstituteMap = std::unordered_map<Expr, Tensor, air::ExprHash, air::NodeEqual>;
class BroadcastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const auto *provide = op->body.as<Provide>();
      CHECK(provide);
      auto call = provide->value.as<Call>();
      CHECK(call);
      BroadcastSet broadcast_args;
      // collect all args which need to insert broadcast.
      broadcast_args = CollectBroadCastArgsFromSpecialOps(call);
      if (broadcast_args.empty()) {
        broadcast_args = CollectBroadCastArgsFromNormalOps(call);
      }

      // if broadcast_args is not empty, then need to insert broadcast
      if (!broadcast_args.empty()) {
        std::vector<Expr> change_args;
        SubstituteMap substitute;
        for (size_t i = 0; i < call->args.size(); ++i) {
          Expr cur_arg = call->args[i];
          if (broadcast_args.count(cur_arg)) {
            // firstly create new tensor with new name
            // then mapping this tensor and need to broadcast arg
            std::string name = "broadcast_" + std::to_string(name_idx_++);
            auto t = placeholder(provide->args, cur_arg.type(), name);
            change_args.emplace_back(cur_arg);
            substitute[cur_arg] = t;
          } else {
            change_args.emplace_back(cur_arg);
          }
        }

        // ChangeAllInputs by substitute and change_args.
        Stmt changed_stmt = ChangeInputs(op, substitute, change_args);
        return DoInsert(op, changed_stmt, substitute, change_args);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Stmt ChangeInputs(const AttrStmt *op, SubstituteMap &substitute, std::vector<Expr> &change_args) {
    const auto *provide = op->body.as<Provide>();
    auto call = provide->value.as<Call>();
    auto args = call->args;
    for (size_t i = 0; i < args.size(); ++i) {
      if (substitute.find(change_args[i]) != substitute.end()) {
        auto t = substitute[change_args[i]];
        args.Set(i, Call::make(t->dtype, t->op->name, t->shape, Call::CallType::Halide, t->op));
      }
    }
    Stmt changed_inputs_stmt = Provide::make(provide->func, provide->value_index,
                                             Call::make(call->type, call->name, args, call->call_type), provide->args);
    Stmt result_stmt = AttrStmt::make(op->node, op->attr_key, op->value, changed_inputs_stmt);
    return result_stmt;
  }

  Stmt DoInsert(const AttrStmt *op, const Stmt &s, SubstituteMap &substitute, std::vector<Expr> &change_args) {
    std::vector<Stmt> stmts;
    for (const auto &arg : change_args) {
      // if has arg and tensor's mapping, then insert broadcast and add attrs.
      if (substitute.find(arg) != substitute.end()) {
        auto t = substitute[arg];
        Map<std::string, NodeRef> attrs = Downcast<Map<std::string, NodeRef>>(op->node);
        attrs.Set("shape", t->shape);
        Stmt broadcast_stmt =
          Provide::make(t->op, 0, Call::make(Int(32), "BroadcastTo", {arg}, Call::CallType::PureIntrinsic), t->shape);
        Stmt broadcast_has_attr_stmt = AttrStmt::make(attrs, "attrs", Expr(1), broadcast_stmt);
        stmts.emplace_back(broadcast_has_attr_stmt);
      }
    }

    stmts.emplace_back(s);
    return Block::make(stmts);
  }

  BroadcastSet CollectBroadCastArgsFromSpecialOps(const Call *call) {
    BroadcastSet broadcast_args;
    auto it = broadcast_ops_.find(call->name);
    // if op is select or equal, find whether args have imm.
    if (it != broadcast_ops_.end()) {
      for (size_t i = 0; i < call->args.size(); ++i) {
        if (!(it->second & (1u << i))) {
          continue;
        }
        Expr e = call->args[i];
        if (e.as<IntImm>() || e.as<UIntImm>() || e.as<FloatImm>()) {
          broadcast_args.insert(e);
          break;
        }
      }
    }
    return broadcast_args;
  }

  BroadcastSet CollectBroadCastArgsFromNormalOps(const Call *call) {
    BroadcastSet broadcast_args;
    bool has_tensor = false;
    if (call->name != "BroadcastTo") {
      for (size_t i = 0; i < call->args.size(); ++i) {
        Expr arg = call->args[i];
        if (arg.as<Call>()) {
          has_tensor = true;
          break;
        }
      }
      if (!has_tensor) {
        // only all args are imm, insert first arg to vector.
        broadcast_args.insert(call->args[0]);
      }
    }
    return broadcast_args;
  }

  int name_idx_ = 0;
  std::unordered_map<std::string, unsigned> broadcast_ops_ = {{"Equal", -1}, {"Select", -1}, {"NotEqual", -1}};
};

Stmt BroadcastInserter(const Stmt &s, BuildInfo *) { return BroadcastInserterMutator().Mutate(s); }
}  // namespace akg
