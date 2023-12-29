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

#include "composite/optimize/pass.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace akg {
namespace {
class RemoveInplaceAssign : public IRMutator {
 public:
  explicit RemoveInplaceAssign(BuildInfo &info) : info_(info) {}
  ~RemoveInplaceAssign() override = default;

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
    CHECK(op->func.defined());
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    for (const auto &arg : call->args) {
      if (args_.find(arg) == args_.end()) {
        args_[arg] = 0;
      }
      args_[arg]++;
    }

    // d = InplaceAssign(a, b, c)
    if (call->name == "InplaceAssign") {
      auto a = call->args[0];
      auto b = call->args[1];
      auto a_call = a.as<Call>();
      auto b_call = b.as<Call>();
      if (IsFakeOutput() && IsOutput(op->func->func_name()) && a_call != nullptr && b_call != nullptr &&
          args_[a] == 1 && IsInput(a_call->name)) {
        // Remove d from output list
        RemoveOutput(op->func);
        // If b is not output, add it to output list (b inplace to a)
        // Else create b_copy = b, and add b_copy to output list (b_copy inplace to a)
        if (!IsOutput(b_call->name)) {
          info_.output_names.push_back(b_call->name);
          info_.opt.inplaces[b_call->func] = a;
        } else {
          auto name = b_call->name + "_copy";
          auto b_copy = placeholder(b_call->args, b.type(), name);
          info_.output_names.push_back(name);
          info_.opt.output_funcs.push_back(b_copy->op);
          info_.opt.sames[b_copy->op] = b_call->func;
          info_.opt.inplaces[b_copy->op] = a;
        }
        return Evaluate::make(0);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  bool IsInput(const std::string &name) {
    return std::find(info_.input_names.begin(), info_.input_names.end(), name) != info_.input_names.end();
  }

  bool IsOutput(const std::string &name) {
    return std::find(info_.output_names.begin(), info_.output_names.end(), name) != info_.output_names.end();
  }

  bool IsFakeOutput() {
    if (op_attrs_.count("fake_output")) {
      auto fake_val = op_attrs_["fake_output"].as<IntImm>();
      if (fake_val && fake_val->value > 0) {
        return true;
      }
    }
    return false;
  }

  void RemoveOutput(const FunctionRef &func) {
    if (!func.defined()) {
      return;
    }
    (void)info_.output_names.erase(std::remove(info_.output_names.begin(), info_.output_names.end(), func->func_name()),
                                   info_.output_names.end());
    (void)info_.opt.output_funcs.erase(std::remove(info_.opt.output_funcs.begin(), info_.opt.output_funcs.end(), func),
                                       info_.opt.output_funcs.end());
  }

  BuildInfo &info_;
  Map<std::string, NodeRef> op_attrs_;
  std::unordered_map<Expr, int, NodeHash, NodeEqual> args_;
};

class FindMissingOutput : public IRVisitor {
 public:
  explicit FindMissingOutput(BuildInfo &info) : info_(info) {}
  ~FindMissingOutput() override = default;

  FuncRefList GetMissingOutput(const Stmt &s) {
    IRVisitor::Visit(s);
    FuncRefList res;
    for (const auto &out_func : info_.opt.output_funcs) {
      if (all_funcs_.find(out_func) == all_funcs_.end()) {
        res.push_back(out_func);
      }
    }
    return res;
  }

 private:
  void Visit_(const Provide *op) override {
    // Collect output
    if (op->func.defined()) {
      all_funcs_.insert(op->func);
    }
    // Collect input
    if (op->value.as<Call>()) {
      auto call = op->value.as<Call>();
      for (const auto &arg : call->args) {
        if (arg.as<Call>() && arg.as<Call>()->func.defined()) {
          all_funcs_.insert(arg.as<Call>()->func);
        }
      }
    }
  }

  BuildInfo &info_;
  FuncRefSet all_funcs_;
};

Stmt AddGlobalAttr(const Stmt &s, const BuildInfo &info) {
  Array<Expr> orig_input_names;
  for (const auto &name : info.input_names) {
    orig_input_names.push_back(Expr(name));
  }
  Array<Expr> orig_output_names;
  for (const auto &name : info.output_names) {
    orig_output_names.push_back(Expr(name));
  }
  Map<std::string, NodeRef> attrs;
  attrs.Set("orig_input_names", orig_input_names);
  attrs.Set("orig_output_names", orig_output_names);
  return AttrStmt::make(attrs, "global_attrs", Expr(1), s);
}
}  // namespace

Stmt ElimInplaceAssign(const Stmt &s, BuildInfo *info) {
  CHECK(info != nullptr);
  // Add global attr, including input names and output names
  auto res = AddGlobalAttr(s, *info);
  // Remove InplaceAssign
  res = RemoveInplaceAssign(*info).Mutate(res);
  // Recover tensors that are real output but not exist in the IR
  auto missing_outputs = FindMissingOutput(*info).GetMissingOutput(res);
  for (const auto &out : missing_outputs) {
    if (info->opt.sames.find(out) != info->opt.sames.end()) {
      auto t = Downcast<Operation>(info->opt.sames[out]).output(0);
      auto t_call = Call::make(t->dtype, info->opt.sames[out]->func_name(), t->shape, Call::CallType::Halide);
      auto p = Provide::make(
        out, 0, Call::make(t->dtype, "Add", {t_call, make_const(t->dtype, 0)}, Call::CallType::PureIntrinsic),
        t->shape);
      res = Block::make(res, p);
    }
  }
  return res;
}
}  // namespace akg
