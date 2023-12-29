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
#include "utils.h"
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include "ir_pass.h"
#include "build_module.h"

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
  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    auto new_op = stmt.as<For>();
    if (new_op != nullptr) {
      bool is_min_binary = IsBinaryOp(new_op->min);
      bool is_max_binary = IsBinaryOp(new_op->extent);
      if (!is_min_binary && is_max_binary) {
        auto extent = new_op->extent.as<Sub>();
        if (extent != nullptr && (extent->a.as<Call>() != nullptr || extent->b.as<Call>() != nullptr)) {
          auto extent_call = extent->a.as<Call>();
          CHECK(extent_call);
          Var max_var = Variable::make(extent_call->type, "MAX_VAR_" + std::to_string(++max_var_count_));
          auto new_stmt = For::make(new_op->loop_var, new_op->min, max_var,
                                    new_op->for_type, new_op->device_api, new_op->body);
          Array<Expr> replaced;
          replaced.push_back(new_op->extent);
          g_csr.Set(max_var, replaced);
          return new_stmt;
        }
      } else if (is_min_binary) {
        LOG(INFO) << "Currently cannot support dynamic shapes in lower bound or both bounds."
                  << "Fall back to original statment.";
      }
    }
    return stmt;
  }

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
      if (IsHalideCall(op->args[i])) {
        lhs_tensor_idx_[op->args[i]] = static_cast<int>(i);
      }
    }

    // remake provide now
    if (!lhs_tensor_idx_.empty()) {
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
    }

    lhs_tensor_idx_.clear();

    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op == nullptr) return e;
    std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> local_rhs_tensor_idx_;

    if (in_args_ && op->call_type == Call::Halide) {
      halide_call_ = true;
      if (cache_idx_.count(op->func.get()) == 0) {
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
        local_rhs_tensor_idx_[op->args[i]] = static_cast<int>(i);
      }
    }
    in_args_ = false;

    // for call not in provide, rhs always
    if (!local_rhs_tensor_idx_.empty()) {
      Array<Expr> idx_args;
      Expr ne = e;

      for (const auto &i : local_rhs_tensor_idx_) {
        int idx = GetTensorIdx(i.first);
        ne = air::ir::substitute(i.first, Expr(idx), ne);
        Expr call = Call::make(op->type, "rhs", {Mutate_(i.first.as<Call>(), i.first), Expr(i.second), Expr(idx)}, 
                               Call::PureIntrinsic);
        idx_args.push_back(call);
      }
      idx_args.push_back(Call::make(op->type, "orig", {ne}, Call::PureIntrinsic));
      local_rhs_tensor_idx_.clear();

      return Call::make(op->type, "with", idx_args, Call::PureIntrinsic);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  int GetTensorIdx(const Expr &v) {
    const Call *op = v.as<Call>();
    if ((op == nullptr) || (cache_idx_.count(op->func.get()) == 0)) {
      if (!IsBinaryOp(v)) {
        has_invalid_tensor_expr_ = true;
        LOG(INFO) << "found invalid tensor expr " << v
                  << " in rewrite_tensor_index, will fallback to rewrite_var_tensor_idx";
        
        return -1;
      }

      return 0;
    }

    return cache_idx_[op->func.get()];
  }

  bool IsBinaryOp(const Expr v) {
    bool res = (v.as<Add>() != nullptr);
    res |= (v.as<Sub>() != nullptr);
    res |= (v.as<Mul>() != nullptr);
    res |= (v.as<Div>() != nullptr);
    res |= (v.as<Mod>() != nullptr);
    res |= (v.as<FloorDiv>() != nullptr);
    res |= (v.as<FloorMod>() != nullptr);
    res |= (v.as<Min>() != nullptr);
    res |= (v.as<Max>() != nullptr);
    return res;
  }

  std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> lhs_tensor_idx_;
  std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> rhs_tensor_idx_;
  std::unordered_map<const Node *, Type> realize_type_;
  Map<FunctionRef, Array<Expr>> max_extent_;
  // in args
  bool in_args_{false};
  bool halide_call_{false};
  int i_{0};
  // for inner tensor index
  std::map<const Node *, int> cache_idx_;
  size_t max_var_count_{0};

 public:
  bool has_invalid_tensor_expr_{false};
};

Stmt RewriteTensorIndex(const Stmt stmt) {
  auto mutator = RewriteTensorIdx();
  auto new_stmt = mutator.Mutate(stmt);

  return mutator.has_invalid_tensor_expr_ ? stmt : new_stmt;
}
}  // namespace ir
}  // namespace akg
