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
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "composite/sync_process.h"
#include "tvm.h"

namespace akg {
namespace ir {
namespace {
constexpr auto ThreadIdx = "threadIdx.";
constexpr auto StorageSync = "tvm_storage_sync";
constexpr auto AkgReduce = "akg_reduce::AkgReduce";
constexpr auto SyncScope = "shared";
constexpr auto ThreadxLen = 10;
}  // namespace

bool ThreadSync(const Call *op) {
  if (op->name == StorageSync) {
    auto scope = op->args[0].as<StringImm>();
    CHECK(scope);
    if (scope->value == SyncScope) {
      return true;
    }
  }
  return false;
}

bool CallNameMatch(const Call *op, const std::string &name) { return op->name == name; }

EvaluateVisitor::EvaluateVisitor() {
  blacklist_call_funcs_ = {std::bind(CallNameMatch, std::placeholders::_1, AkgReduce)};
  target_call_func_ = ThreadSync;
}

std::pair<bool, bool> EvaluateVisitor::Run(const Stmt &stmt) {
  Visit(stmt);
  return std::make_pair(target_hit_, blacklist_hit_);
}

void EvaluateVisitor::Visit_(const Call *op) {
  IRVisitor::Visit_(op);
  if (in_evaluate_) {
    if (target_call_func_(op)) {
      target_hit_ = true;
    }

    if (std::any_of(blacklist_call_funcs_.begin(), blacklist_call_funcs_.end(),
                    [&op](const std::function<bool(const Call *op)> &f) { return f(op); })) {
      blacklist_hit_ = true;
    }
  }
}

void EvaluateVisitor::Visit_(const Evaluate *op) {
  in_evaluate_ = true;
  IRVisitor::Visit(op->value);
  in_evaluate_ = false;
}

class SyncProcess : public IRMutator {
 public:
  std::pair<bool, Stmt> Run(const Stmt &stmt) {
    auto res_stmt = Mutate(stmt);
    res_stmt = CanonicalSimplify(res_stmt);
    return std::make_pair(bad_transform_, res_stmt);
  }

 private:
  Stmt WrapWithIfStack(const Stmt &s) {
    Stmt res_stmt = s;
    std::stack<std::pair<Expr, bool>> stack_tmp(if_cond_stack_);
    // IfThenElse::make
    while (!stack_tmp.empty()) {
      auto cond_pair = stack_tmp.top();
      stack_tmp.pop();
      if (cond_pair.second) {
        res_stmt = IfThenElse::make(cond_pair.first, res_stmt);
      } else {
        res_stmt = IfThenElse::make(cond_pair.first, Evaluate::make(0), res_stmt);
      }
    }
    return res_stmt;
  }

  Expr Mutate_(const Variable *op, const Expr &e) override {
    if (in_cond_ && op->name_hint.compare(0, ThreadxLen, ThreadIdx) == 0) {
      in_thread_cond_ = true;
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (ThreadSync(op)) {
      call_shared_sync_ = true;
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) override {
    auto res_stmt = IRMutator::Mutate_(op, s);
    if (exist_nasty_sync_) {
      res_stmt = WrapWithIfStack(res_stmt);
    }
    return res_stmt;
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) override {
    call_shared_sync_ = false;
    auto res_stmt = IRMutator::Mutate_(op, s);
    if (exist_nasty_sync_ && !call_shared_sync_) {
      res_stmt = WrapWithIfStack(res_stmt);
    }
    return res_stmt;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) override {
    std::pair<bool, bool> evaluate_res;
    if (if_cond_stack_.empty()) {
      // Check whether these is sync in children block or not.
      evaluate_res = EvaluateVisitor().Run(s);
    }

    // Check whether this is a thread cond IfThenElse or not.
    in_cond_ = true;
    in_thread_cond_ = false;
    Expr cond = Mutate(op->condition);
    in_cond_ = false;

    Stmt res_stmt;
    if (in_thread_cond_) {
      if (if_cond_stack_.empty()) {
        exist_nasty_sync_ = evaluate_res.first;
        if (evaluate_res.second) {
          LOG(WARNING) << "Exist bad stmt in thread condition!";
          bad_transform_ = true;
        }
      }
      std::vector<Stmt> stmts;
      if_cond_stack_.push(std::make_pair(op->condition, true));
      auto first_stmt = Mutate(op->then_case);
      if_cond_stack_.pop();
      stmts.emplace_back(first_stmt);
      Stmt second_stmt = op->else_case;
      if (op->else_case.defined()) {
        if_cond_stack_.push(std::make_pair(op->condition, false));
        second_stmt = Mutate(op->else_case);
        if_cond_stack_.pop();
        stmts.emplace_back(second_stmt);
      }
      res_stmt = IfThenElse::make(op->condition, first_stmt, second_stmt);
      if (exist_nasty_sync_) {
        res_stmt = Block::make(stmts);
      }
    } else {
      res_stmt = IRMutator::Mutate_(op, s);
    }
    in_thread_cond_ = false;

    return res_stmt;
  }

  bool bad_transform_{false};
  bool in_cond_{false};
  bool in_thread_cond_{false};
  bool exist_nasty_sync_{false};
  bool call_shared_sync_{false};
  std::stack<std::pair<Expr, bool>> if_cond_stack_;  // stack for condition of IfThenElse which contain threadidx...
};

Stmt ProcessSyncInnerThread(const Stmt &stmt) {
  Stmt res_stmt;
  bool bad_transform;
  std::tie(bad_transform, res_stmt) = SyncProcess().Run(stmt);
  if (bad_transform) {
    return stmt;
  }
  return res_stmt;
}
}  // namespace ir
}  // namespace akg
