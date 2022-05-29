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

namespace akg {
class FindProvide : public IRVisitor {
 public:
  explicit FindProvide(const FunctionRef &func) : func_(func) {}
  void Visit_(const Provide *op) override {
    if (op->func == func_) provide_ = op;
  }
  const FunctionRef func_;
  const Provide *provide_{nullptr};
};

bool IsOutput(const FuncRefList &outputs, const FunctionRef &func) {
  return std::find(outputs.begin(), outputs.end(), func) != outputs.end();
};

class NormalizeOutput : public IRMutator {
 public:
  NormalizeOutput(BuildInfo *info, const std::vector<FuncRefSet> &sames_vec)
      : info_(info), outputs_(info->opt.output_funcs), sames_vec_(sames_vec){};

 private:
  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    if (!IsOutput(outputs_, op->func)) {
      for (const auto &i : sames_vec_) {
        if (i.count(op->func)) {
          for (const auto &j : i) {
            if (IsOutput(outputs_, j)) {
              replaces_[op->func] = j;
              auto it = info_->opt.inplaces.find(op->func);
              if (it != info_->opt.inplaces.end()) {
                info_->opt.inplaces[j] = info_->opt.inplaces[op->func];
                info_->opt.inplaces.erase(it);
              }
              return Provide::make(j, op->value_index, this->Mutate(op->value), op->args);
            }
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args;
    for (const auto &arg : op->args) {
      if (auto tensor = arg.as<Call>()) {
        if (replaces_.count(tensor->func)) {
          auto replaced = replaces_[tensor->func];
          args.push_back(Call::make(tensor->type, replaced->func_name(), tensor->args, tensor->call_type, replaced,
                                    tensor->value_index));
        } else {
          args.push_back(arg);
        }
      } else {
        args.push_back(arg);
      }
    }
    return Call::make(op->type, op->name, args, op->call_type, op->func);
  }
  BuildInfo *info_;
  FuncRefList outputs_;
  FuncRefMap replaces_;
  std::vector<FuncRefSet> sames_vec_;
};

std::vector<FuncRefSet> NormalizeSames(const FuncRefMap &sames) {
  std::vector<FuncRefSet> sames_vec;
  for (const auto &[k, v] : sames) {
    bool found = false;
    for (auto &i : sames_vec) {
      if (i.count(k) || i.count(v)) {
        i.insert(k);
        i.insert(v);
        found = true;
      }
    }
    if (found) continue;
    sames_vec.push_back({k, v});
  }
  return sames_vec;
}

Stmt BroadcastForSSA(const Stmt &s, BuildInfo *info) {
  auto outputs = info->opt.output_funcs;
  std::vector<FuncRefSet> sames_vec = NormalizeSames(info->opt.sames);
  auto stmt = NormalizeOutput(info, sames_vec).Mutate(s);
  std::vector<Stmt> stmts;
  stmts.push_back(stmt);
  for (auto &i : sames_vec) {
    for (const auto &j : i) {
      auto find = FindProvide(j);
      find.Visit(stmt);
      if (IsOutput(outputs, j) && find.provide_) {
        for (const auto &k : i) {
          if (j != k && IsOutput(outputs, k)) {
            auto shape = find.provide_->args;
            CHECK((find.provide_->value).as<Call>());
            auto type = (find.provide_->value).as<Call>()->type;
            auto input_call = Call::make(type, j->func_name(), shape, Call::CallType::Halide, j);
            auto p =
              Provide::make(k, 0, Call::make(type, "BroadcastTo", {input_call}, Call::CallType::PureIntrinsic), shape);
            Map<std::string, NodeRef> broad_attrs;
            broad_attrs.Set("shape", shape);
            p = AttrStmt::make(broad_attrs, "attrs", Expr(1), p);
            stmts.push_back(p);
          }
        }
      }
    }
  }
  return Block::make(stmts);
}
}  // namespace akg
