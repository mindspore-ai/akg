/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include <pass/utils.h>
#include "ir_pass.h"

namespace akg {
namespace ir {
/*
Add attributes for layout operators.
*/

class AttrForLayoutOp : public IRMutator {
 public:
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    CHECK(op);
    if (IsCountOp(op)) {
      stmt = AddAttrForAtomicToT(stmt.as<Provide>(), op, stmt);
    } else if (ContainsHalideCall(op->args)) {
      is_tensor_of_tensor_ = true;
      tensors_not_promote_.insert(op->func->func_name());
      if (CheckBinaryCall(op)) {
        stmt = AddAttrForAtomicToT(stmt.as<Provide>(), op, stmt);
      }
    }
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    CHECK(op);
    // If we have gone through the outermost tensor, and the current call
    // is in format of `input_x()`, then current tensor is an inner tensor.
    if (in_args_ && op->call_type == Call::Halide && halide_call_) {
      inner_tensors_.insert(op->func->func_name());
    }
    in_args_ = true;
    for (size_t i = 0; i < op->args.size(); ++i) {
      halide_call_ = (op->call_type == Call::Halide);
      static_cast<void>(this->Mutate(op->args[i]));
    }
    // If current call is in format of `input_x()`, and its args are also
    // in format of `input_x()`, then the current tensor is a tensor of tensor
    if (op->call_type == Call::Halide && ContainsHalideCall(op->args)) {
      is_tensor_of_tensor_ = true;
      tensors_not_promote_.insert(op->func->func_name());
    }
    in_args_ = false;
    return e;
  }

 private:
  Stmt AddAttrForAtomicToT(const Provide *new_op, const Provide *op, Stmt stmt) {
    return AttrStmt::make(new_op->func, "atomic_tot", Expr(GetOpReduceType(op->value)), stmt);
  }

  bool ContainsCSR(std::string name) {
    return name.find("csr") != std::string::npos;
  }

  bool CheckBinaryCall(const Provide* op) {
    CHECK(op);
    if (GetOpReduceType(op->value) != AKG_REDUCE_UNSUPPORTED) {
      auto array = GetBinaryOpExprChildren(op->value);
      auto func_name = op->func->func_name();
      auto call_a = array[0].as<Call>();
      auto call_b = array[1].as<Call>();
      if (call_a && call_b && (call_a->name == func_name || call_b->name == func_name)) return true;
    }
    return ContainsHalideCall(op->args);
  }

  // in args
  bool in_args_{false};
  bool halide_call_{false};

 public:
  std::unordered_set<std::string> tensors_not_promote_;
  std::unordered_set<std::string> inner_tensors_;
  bool is_tensor_of_tensor_{false};
};

class RemoveLikely : public IRMutator {
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->is_intrinsic(Call::likely)) {
      CHECK(op->args.size() > 0);
      return IRMutator::Mutate(op->args[0]);
    }
    return IRMutator::Mutate_(op, e);
  }
};

Stmt AddAttrForLayoutOp(const Stmt stmt, Schedule sch, bool need_remove_likely) {
  auto mutator = AttrForLayoutOp();
  auto new_stmt = mutator.Mutate(stmt);
  if (!mutator.tensors_not_promote_.empty()) {
    for (auto &t : mutator.tensors_not_promote_) {
      new_stmt = AttrStmt::make(Expr("INFO"), AKG_TENSOR_NOT_PROMOTE, Expr(t), new_stmt);
    }
  }

  if (!mutator.inner_tensors_.empty()) {
    for (auto &t : mutator.inner_tensors_) {
      new_stmt = AttrStmt::make(Expr("INFO"), AKG_INNER_TENSOR, Expr(t), new_stmt);
    }
  }

  if (mutator.is_tensor_of_tensor_) {
    new_stmt = AttrStmt::make(Expr("INFO"), AKG_TENSOR_OF_TENSOR, Expr(AKG_TENSOR_OF_TENSOR), new_stmt);
  }

  if (AttrExists(sch, "remove_self_dependence")) {
    new_stmt = AttrStmt::make(Expr("INFO"), AKG_REMOVE_SELF_DEPENDENCE, Expr(AKG_REMOVE_SELF_DEPENDENCE), new_stmt);
  }

  if (need_remove_likely) {
    new_stmt = RemoveLikely().Mutate(new_stmt);
  }
  return new_stmt;
}
}  // namespace ir
}  // namespace akg
