/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

#include <regex>

#include "ir_pass.h"
#include "pass/utils.h"

namespace akg {
namespace ir {

class DivModMutator : public IRMutator {
 private:
  Expr Mutate_(const Div *op, const Expr &e) final {
    if (op->b.as<IntImm>() && op->b.as<IntImm>()->value > 1) {
      auto value = static_cast<uint16_t>(op->b.as<IntImm>()->value);
      if (!(value & (value - 1))) {
        return op->a >> Log2(value);
      }
    }
    return e;
  }

  Expr Mutate_(const FloorDiv *op, const Expr &e) final {
    if (op->b.as<IntImm>() && op->b.as<IntImm>()->value > 1) {
      auto value = static_cast<uint16_t>(op->b.as<IntImm>()->value);
      if (!(value & (value - 1))) {
        return op->a >> Log2(value);
      }
    }
    return e;
  }

  Expr Mutate_(const Mod *op, const Expr &e) final {
    if (op->b.as<IntImm>() && op->b.as<IntImm>()->value > 1) {
      auto value = static_cast<uint16_t>(op->b.as<IntImm>()->value);
      if (!(value & (value - 1))) {
        return op->a & (value - 1);
      }
    }
    return e;
  }

  Expr Mutate_(const FloorMod *op, const Expr &e) final {
    if (op->b.as<IntImm>() && op->b.as<IntImm>()->value > 1) {
      auto value = static_cast<uint16_t>(op->b.as<IntImm>()->value);
      if (!(value & (value - 1))) {
        return op->a & (value - 1);
      }
    }
    return e;
  }
};

class ConvertShiftMutator : public IRMutator {
 private:
  Expr ShiftMutator(const Expr &e) const { return DivModMutator().Mutate(e); }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Expr min = ShiftMutator(op->min);
    Expr extent = ShiftMutator(op->extent);
    
    Stmt stmt;
    stmt = For::make(op->loop_var, min, extent, op->for_type, op->device_api, Mutate(op->body));
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args = op->args;
    for (auto i = 0u; i < args.size(); ++i) {
      args.Set(i, ShiftMutator(args[i]));
    }
    return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Array<Expr> args = op->args;
    for (auto i = 0u; i < args.size(); ++i) {
      args.Set(i, ShiftMutator(args[i]));
    }
    return Provide::make(op->func, op->value_index, Mutate(op->value), args);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    return Load::make(op->type, op->buffer_var, ShiftMutator(op->index), op->predicate);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    return Store::make(op->buffer_var, Mutate(op->value), ShiftMutator(op->index), op->predicate);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Array<Expr> extents;
    for (auto &extent : op->extents) {
      extents.push_back(ShiftMutator(extent));
    }
    return Allocate::make(op->buffer_var, op->type, extents, op->condition, Mutate(op->body),
                          (op->new_expr.defined() ? ShiftMutator(op->new_expr) : op->new_expr), op->free_function);
  }
};
Stmt ConvertDivModToShift(const Stmt &stmt) { return ConvertShiftMutator().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
