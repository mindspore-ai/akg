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
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include "ir_pass.h"

namespace akg {
namespace ir {
/*
Rewrite tensor of tensor indexes as a special "with" representation.

Left-hand side: before:

realize one_hot_hybrid([0, 16], [0, 30522]) {
  for (z, 0, 16) {
    one_hot_hybrid(z, input_1(z)) = 1
  }
}

after:

realize one_hot_hybrid([0, 16], [0, 30522]) {
  for (z, 0, 16) {
     one_hot_hybrid_1(i, 30521) = with(lhs(input_1(z), 1, 30521), orig(1))
  }
}

Right-hand side: before:

realize one_hot_hybrid([0, 16], [0, 30522]) {
  realize result([0, 16]) {
    for (z, 0, 16) {
      result(z) = one_hot_hybrid(z, input_1(z))
    }
  }
}

after:

realize one_hot_hybrid([0, 16], [0, 30522]) {
  realize result([0, 16]) {
    for (z, 0, 16) {
      result(z) = with(rhs(input_1(z), 0, 0), orig(one_hot_hybrid_2(z, 30521)))
    }
  }
}
*/

class RewriteTensorIdx : public IRMutator {
 public:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    std::vector<Expr> extents;
    for (const auto &i : op->bounds) {
      extents.push_back(Simplify_cce(i->extent - 1));
    }
    max_extent_.Set(op->func, extents);
    realize_type_[op->func.get()] = op->type;

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    lhs_tensor_idx_.clear();
    rhs_tensor_idx_.clear();
    Stmt stmt = IRMutator::Mutate_(op, s);
    for (size_t i = 0; i < op->args.size(); ++i) {
      const Call *call = op->args[i].as<Call>();
      if (call && call->call_type == Call::Halide) {
        lhs_tensor_idx_[op->args[i]] = static_cast<int>(i);
      }
    }

    // remake provide now
    if (!lhs_tensor_idx_.empty()) {
      is_tensor_of_tensor_ = true;
      tensors_not_promote_.insert(op->func->func_name());
      // build a new value
      Array<Expr> idx_args;
      Expr extent = Expr(0);
      Type type_ = Int(32);
      std::map<int, Expr> rewrite_args;

      for (const auto &i : lhs_tensor_idx_) {
        if (max_extent_.count(op->func)) {
          extent = max_extent_[op->func][i.second];
          type_ = realize_type_[op->func.get()];
        }
        rewrite_args[i.second] = extent;

        Expr call = Call::make(type_, "lhs", {i.first, Expr(i.second), extent}, Call::PureIntrinsic);
        idx_args.push_back(call);
      }

      // update args, replace tensor index with imm val
      const auto new_op = stmt.as<Provide>();
      Array<Expr> new_args;
      CHECK(new_op != nullptr);
      for (size_t i = 0; i < new_op->args.size(); ++i) {
        if (rewrite_args.count(i)) {
          new_args.push_back(rewrite_args[i]);
        } else {
          new_args.push_back(new_op->args[i]);
        }
      }

      idx_args.push_back(Call::make(type_, "orig", {new_op->value}, Call::PureIntrinsic));
      Expr val = Call::make(type_, "with", idx_args, Call::PureIntrinsic);
      stmt = Provide::make(new_op->func, new_op->value_index, val, new_args);
      stmt = AddAttrForAtomicToT(new_op, op, stmt);
    }

    lhs_tensor_idx_.clear();

    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (in_args_ && op->call_type == Call::Halide) {
      halide_call_ = true;
      if (cache_idx_.count(op->func.get()) == 0) {
        inner_tensors_.insert(op->func->func_name());
        cache_idx_[op->func.get()] = i_;
        i_ = i_ + 2;
      }

      return e;
    }

    in_args_ = true;
    for (size_t i = 0; i < op->args.size(); ++i) {
      halide_call_ = false;
      static_cast<void>(this->Mutate(op->args[i]));
      if (op->call_type == Call::Halide && halide_call_) {
        rhs_tensor_idx_[op->args[i]] = static_cast<int>(i);
      }
    }
    in_args_ = false;

    // for call not in provide, rhs always
    if (!rhs_tensor_idx_.empty()) {
      is_tensor_of_tensor_ = true;
      tensors_not_promote_.insert(op->func->func_name());
      Array<Expr> idx_args;
      Expr ne = e;

      for (const auto &i : rhs_tensor_idx_) {
        int idx = GetTensorIdx(i.first);
        ne = air::ir::substitute(i.first, Expr(idx), ne);
        Expr call = Call::make(op->type, "rhs", {i.first, Expr(i.second), Expr(idx)}, Call::PureIntrinsic);
        idx_args.push_back(call);
      }
      idx_args.push_back(Call::make(op->type, "orig", {ne}, Call::PureIntrinsic));
      rhs_tensor_idx_.clear();

      return Call::make(op->type, "with", idx_args, Call::PureIntrinsic);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  int GetTensorIdx(const Expr &v) {
    const Call *op = v.as<Call>();
    if ((op == nullptr) || (cache_idx_.count(op->func.get()) == 0)) {
      has_invalid_tensor_expr_ = true;
      LOG(INFO) << "found invalid tensor expr " << v
                << " in rewrite_tensor_index, will fallback to rewrite_var_tensor_idx";

      return -1;
    }

    return cache_idx_[op->func.get()];
  }

  Stmt AddAttrForAtomicToT(const Provide *new_op, const Provide *op, Stmt stmt) {
    auto Get = [new_op, op, stmt](const Expr a, const Expr b, std::string op_type) -> Stmt {
      auto func_name = op->func->func_name();
      auto call_a = a.as<Call>();
      auto call_b = b.as<Call>();
      if (call_a && call_b && (call_a->name == func_name || call_b->name == func_name)) {
        return AttrStmt::make(new_op->func, "atomic_tot", Expr(op_type), stmt);
      }
      return stmt;
    };

    if (auto atomic_op = op->value.as<Max>()) {
      stmt = Get(atomic_op->a, atomic_op->b, "MaxOp");
    } else if (auto atomic_op = op->value.as<Min>()) {
      stmt = Get(atomic_op->a, atomic_op->b, "MinOp");
    } else if (auto atomic_op = op->value.as<And>()) {
      stmt = Get(atomic_op->a, atomic_op->b, "AndOp");
    } else if (auto atomic_op = op->value.as<Or>()) {
      stmt = Get(atomic_op->a, atomic_op->b, "OrOp");
    } else if (auto atomic_op = op->value.as<Add>()) {
      stmt = Get(atomic_op->a, atomic_op->b, "SumOp");
    }

    return stmt;
  }

  std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> lhs_tensor_idx_;
  std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> rhs_tensor_idx_;
  std::unordered_map<const Node *, Type> realize_type_;
  Map<FunctionRef, Array<Expr>> max_extent_;
  // in args
  bool in_args_{false};
  bool halide_call_{false};
  Type type;
  int i_{0};
  // for inner tensor index
  std::map<const Node *, int> cache_idx_;

 public:
  bool has_invalid_tensor_expr_{false};
  std::unordered_set<std::string> tensors_not_promote_;
  std::unordered_set<std::string> inner_tensors_;
  bool is_tensor_of_tensor_{false};
};

Stmt RewriteTensorIndex(const Stmt stmt) {
  auto mutator = RewriteTensorIdx();
  auto new_stmt = mutator.Mutate(stmt);
  if (!mutator.tensors_not_promote_.empty()) {
    for (auto &t : mutator.tensors_not_promote_) {
      new_stmt = AttrStmt::make(Expr("INFO"), "TENSOR_NOT_PROMOTE", Expr(t), new_stmt);
    }
  }

  if (!mutator.inner_tensors_.empty()) {
    for (auto &t : mutator.inner_tensors_) {
      new_stmt = AttrStmt::make(Expr("INFO"), "INNER_TENSOR", Expr(t), new_stmt);
    }
  }

  if (mutator.is_tensor_of_tensor_) {
    new_stmt = AttrStmt::make(Expr("INFO"), "TENSOR_OF_TENSOR", Expr("TENSOR_OF_TENSOR"), new_stmt);
  }

  return mutator.has_invalid_tensor_expr_ ? stmt : new_stmt;
}
}  // namespace ir
}  // namespace akg
