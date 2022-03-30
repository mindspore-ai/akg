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

Stmt BroadcastForSSA(const Stmt &s, BuildInfo *info) {
  auto &sames = info->opt.sames;
  auto outputs = info->opt.output_funcs;
  auto IsOutput = [&outputs](const FunctionRef &func) {
    return std::find(outputs.begin(), outputs.end(), func) != outputs.end();
  };
  auto HasSameIn = [&sames](const FunctionRef &in, const FunctionRef &out) {
    bool same = false;
    for (const auto &[first, second] : sames) {
      if (second == in && first != out) same = true;
    }
    return same;
  };
  std::vector<Stmt> stmts;
  stmts.push_back(s);
  for (auto it = sames.begin(); it != sames.end();) {
    auto out = it->first;
    auto in = it->second;
    if (IsOutput(out) && (IsOutput(in) || HasSameIn(in, out))) {
      auto find = FindProvide(in);
      find.Visit(s);
      if (find.provide_) {
        auto shape = find.provide_->args;
        CHECK((find.provide_->value).as<Call>());
        auto type = (find.provide_->value).as<Call>()->type;
        auto input_call = Call::make(type, in->func_name(), shape, Call::CallType::Halide, in);
        auto stmt =
          Provide::make(out, 0, Call::make(type, "BroadcastTo", {input_call}, Call::CallType::PureIntrinsic), shape);
        Map<std::string, NodeRef> broad_attrs;
        broad_attrs.Set("shape", shape);
        stmt = AttrStmt::make(broad_attrs, "attrs", Expr(1), stmt);
        stmts.push_back(stmt);
        it = sames.erase(it);
        continue;
      }
    }
    ++it;
  }
  return Block::make(stmts);
}
}  // namespace akg
