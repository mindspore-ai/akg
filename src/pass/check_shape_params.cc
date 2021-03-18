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
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include "pass/utils.h"

namespace akg {
namespace ir {
class CheckShapeParamsMutator : public IRMutator {
 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    bool need_shift = false;
    auto shifted_bounds = op->bounds;
    unsigned int num_dim = op->bounds.size();
    for (unsigned int i = 0; i < num_dim; ++i) {
      auto dim_bound = op->bounds[i];
      CHECK(dim_bound->min.as<IntImm>()) << "min of realize shape must be an IntImm,"
                                         << " but found " << op->bounds << " in tensor " << op->func->func_name();
      if (dim_bound->min.as<IntImm>()->value != 0) {
        need_shift = true;
        shifted_bounds.Set(i, Range::make_by_min_extent(0, dim_bound->extent));
      }
      if (dim_bound->extent.as<IntImm>() != nullptr && dim_bound->extent.as<IntImm>()->value <= 0) {
        CHECK(0) << "realize shape must have extent > 0, but found " << op->bounds << " in tensor "
                 << op->func->func_name();
      }
    }

    if (need_shift) {
      tensors_need_shift[op->func] = op;
      Stmt body = Mutate(op->body);
      tensors_need_shift.erase(op->func);
      return Realize::make(op->func, op->value_index, op->type, shifted_bounds, op->condition, body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    if (op->extent.as<IntImm>() != nullptr && op->extent.as<IntImm>()->value <= 0) {
      return Evaluate::make(0);
    }
    return IRMutator::Mutate_(op, s);
  }

  template <class T>
  Array<Expr> ShiftCallArgs(const T *op) {
    auto new_call_args = op->args;
    const Realize *realize = tensors_need_shift.at(op->func);
    unsigned int num_dim = realize->bounds.size();
    CHECK(new_call_args.size() == num_dim) << "dims in call and realize mismatch";
    for (unsigned int i = 0; i < num_dim; ++i) {
      auto dim_bound = realize->bounds[i]->min.as<IntImm>();
      CHECK(dim_bound != nullptr);
      auto dim_bound_int_val = dim_bound->value;
      if (dim_bound_int_val != 0) {
        new_call_args.Set(i, Simplify(op->args[i] - realize->bounds[i]));
      }
    }
    return new_call_args;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    Stmt stmt = IRMutator::Mutate_(op, s);
    auto new_op = stmt.as<Provide>();
    CHECK(new_op != nullptr);
    if (tensors_need_shift.count(new_op->func) > 0) {
      return Provide::make(new_op->func, new_op->value_index, new_op->value, ShiftCallArgs(new_op));
    } else {
      return stmt;
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    Expr expr = IRMutator::Mutate_(op, e);
    auto new_op = expr.as<Call>();
    CHECK(new_op != nullptr);
    if (tensors_need_shift.count(new_op->func) > 0) {
      return Call::make(new_op->type, new_op->name, ShiftCallArgs(new_op), new_op->call_type, new_op->func,
                        new_op->value_index);
    } else {
      return expr;
    }
  }

  std::unordered_map<FunctionRef, const Realize *, NodeHash, NodeEqual> tensors_need_shift;
};

void CheckExternBuffers(const Map<Tensor, Buffer> &extern_buffer) {
  for (auto buffer : extern_buffer) {
    for (auto dim_size : buffer.first->shape) {
      if (dim_size.as<IntImm>() != nullptr && dim_size.as<IntImm>()->value <= 0) {
        CHECK(0) << "dim size must be positive, but found " << buffer.first;
      }
    }
    for (auto dim_size : buffer.second->shape) {
      if (dim_size.as<IntImm>() != nullptr && dim_size.as<IntImm>()->value <= 0) {
        CHECK(0) << "dim size must be positive, but found " << buffer.second;
      }
    }
  }
}

Stmt CheckShapeParams(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  CheckExternBuffers(extern_buffer);
  return CheckShapeParamsMutator().Mutate(std::move(stmt));
}
}  // namespace ir
}  // namespace akg
