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

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm.h>
#include "ir_pass.h"

namespace akg {
namespace ir {
namespace {
/* Visits the Stmt to find loops with non-constant extents. */
class NonConstantExtFinder : public IRVisitor {
 public:
  void Visit_(const Variable *ex) final {
    if (find_var) {
      found_var = ex;
    }
  }

  void Visit_(const For *op) final {
    if (mutate) return;

    visited_loops.push_back(op);

    if (!is_const(op->extent)) {
      auto extent = op->extent;

      find_var = true;
      Visit(extent);
      find_var = false;

      if (found_var != nullptr) {
        int item = GetLoopOfVar(found_var->name_hint);
        if (item != -1) {
          unrollCand = visited_loops[item];
          mutate = true;
        }
      }
    }

    Visit(op->body);
    visited_loops.pop_back();
  }

  int GetLoopOfVar(const std::string &var_hint) {
    for (int i = static_cast<int>(visited_loops.size()) - 1; i >= 0; --i) {
      if (visited_loops[i]->loop_var->name_hint == var_hint) {
        return i;
      }
    }

    return -1;
  }

  /* Auxiliary function to restart everything and start finding loops again */
  void restart() {
    mutate = false;
    visited_loops = std::vector<const For *>();
    find_var = false;
  }

  bool Mutate() const { return mutate; }
  const For *getUnrollCandidate() const { return unrollCand; }

 private:
  bool find_var{false};
  const Variable *found_var{nullptr};
  bool mutate{false};
  std::vector<const For *> visited_loops;
  const For *unrollCand{nullptr};
};

/* Class that unrolls a loop which loop_var is being using as part of an extent of an internal loop */
class NonConstantExtentUnroller : public IRMutator {
 public:
  NonConstantExtentUnroller() : loopFinder(), replace_value(0) {}
  ~NonConstantExtentUnroller() override = default;

  Stmt VisitAndMutate(Stmt stmt) {
    loopFinder.Visit(stmt);
    while (loopFinder.Mutate()) {
      auto ret = Mutate(stmt);
      if (ret.same_as(stmt)) break;

      stmt = ret;
      loopFinder.restart();
      loopFinder.Visit(stmt);
    }

    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op == loopFinder.getUnrollCandidate()) {
      auto min = op->min.as<IntImm>();
      auto extent = op->extent.as<IntImm>();
      Stmt new_loop;

      if (min && extent) {
        std::vector<Stmt> inner_loops;
        for (int64_t i = min->value; i < min->value + extent->value; ++i) {
          replace_value = static_cast<int>(i);
          auto ret = air::ir::Substitute(op->body, {{Var{op->loop_var}, make_const(Int(32), replace_value)}});
          ret = Simplify_cce(ret);
          inner_loops.push_back(ret);
        }
        new_loop = Block::make(inner_loops);
      }

      if (new_loop.defined()) {
        return new_loop;
      }
    }

    auto body = Mutate(op->body);
    if (!body.same_as(op->body)) {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }

    return s;
  }

 private:
  NonConstantExtFinder loopFinder;
  int replace_value;
};
}  // namespace

Stmt UnrollNonConstantExtent(const Stmt s) { return NonConstantExtentUnroller().VisitAndMutate(s); }
}  // namespace ir
}  // namespace akg
