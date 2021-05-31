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
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include <stack>
#include "tvm.h"

namespace akg {
namespace ir {
class AddImmMatcher : public IRMutator {
 public:
  AddImmMatcher() {}
  ~AddImmMatcher() override = default;

  bool AllZero(const Array<Expr> &args) {
    for (auto i : args) {
      if (i.as<IntImm>() == nullptr) {
        return false;
      }
    }

    return true;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() != nullptr &&
        op->value.as<StringImm>()->value == "elewise_binary_Add_imm") {
      in_elws_add_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      in_elws_add_ = false;

      // if one src name is the same as dst
      // and the dst idx is the first idx with another
      // this is a last reduce, modify attr as vcadd
      // eg: output_local_UB(0) =(output_local_UB(0) + input1_local_UB(cc2))
      bool same_name = false;
      for (auto i : src_) {
        const auto call = i.as<Call>();
        if (call != nullptr && dst_name_ == call->name && dst_axis_.size() == call->args.size() &&
            AllZero(call->args) && AllZero(dst_axis_)) {
          same_name = true;
          break;
        }
      }

      if (same_name) {
        stmt = AttrStmt::make(op->node, op->attr_key, StringImm::make("elewise_binary_Add"), op->body);
      }

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (in_elws_add_) {
      dst_axis_ = op->args;
      dst_name_ = op->func->func_name();
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (in_elws_add_) {
      src_.push_back(e);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  bool in_elws_add_{false};
  Array<Expr> dst_axis_;
  Array<Expr> src_;
  std::string dst_name_;
};

class ScatterMatcher : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    bool is_scatter_type =
      ((op->value.as<StringImm>() != nullptr) &&
       (op->value.as<StringImm>()->value == "broadcast" || op->value.as<StringImm>()->value == "dma_copy"));
    if (is_scatter_type && (op->attr_key == "pragma_emit_insn")) {
      matched_stack_.push(false);
      Stmt ret = IRMutator::Mutate_(op, s);
      const auto ret_op = ret.as<AttrStmt>();
      bool is_scatter_matched = matched_stack_.top();
      matched_stack_.pop();

      if (is_scatter_matched == true) {
        CHECK(ret_op);
        return AttrStmt::make(ret_op->node, ret_op->attr_key, StringImm::make("scatter"), ret_op->body);
      } else {
        return ret;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (matched_stack_.empty()) {
      return s;
    }

    for (size_t i = 0; i < op->args.size(); ++i) {
      const auto arg_call = op->args[i].as<Call>();
      if ((arg_call != nullptr) && (arg_call->call_type == Call::Halide)) {
        matched_stack_.top() = true;
      }
    }

    Expr src = op->value;
    if (src.as<Call>()) {
      const auto src_call = src.as<Call>();
      for (size_t i = 0; i < src_call->args.size(); i++) {
        const auto arg_call = src_call->args[i].as<Call>();
        if ((arg_call != nullptr) && (arg_call->call_type == Call::Halide)) {
          matched_stack_.top() = true;
        }
      }
    }

    return s;
  }

 private:
  std::stack<bool> matched_stack_;
};

Stmt InjectAttr(Stmt stmt) {
  stmt = AddImmMatcher().Mutate(stmt);
  stmt = ScatterMatcher().Mutate(stmt);

  return stmt;
}
}  // namespace ir
}  // namespace akg
