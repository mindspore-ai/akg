/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tvm.h"

namespace akg {
namespace ir {
/**
 *
 * for (ee0, 0, (LEN - 1) / TILE + 1) {
 *   allocate buff[min(TILE, LEN - ee0 * TILE)]
 * }
 *
 * -->
 *
 * for (ee0, 0, (LEN - 1) / TILE + 1) {
 *   allocate buff[TILE]
 * }
 *
 */
class AllocateUnify : public IRMutator {
 public:
  AllocateUnify() {}
  ~AllocateUnify() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    lvMap_.emplace(std::pair<std::string, VarExpr>{name, var});
    Stmt stmt = IRMutator::Mutate_(op, s);
    lvMap_.erase(name);

    return stmt;
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Stmt stmt;

    Array<Expr> extents = op->extents;
    for (size_t i = 0; i < op->extents.size(); ++i) {
      Expr ext = op->extents[i];

      if (MaxBoundExpr(ext)) {
        CHECK(ext.as<Min>());
        extents.Set(i, ext.as<Min>()->a);
      }
    }

    return Allocate::make(op->buffer_var, op->type, extents, op->condition, op->body);
  }

 private:
  // Min(T, expr - var * T)
  bool MaxBoundExpr(Expr e) {
    auto min = e.as<Min>();
    if (min == nullptr) {
      return false;
    }

    auto body = min->a.as<Variable>();
    auto tail = min->b.as<Sub>();
    if (body == nullptr || tail == nullptr) {
      return false;
    }

    auto mul = tail->b.as<Mul>();
    if (mul == nullptr) {
      return false;
    }

    auto varA = mul->a.as<Variable>();
    auto varB = mul->a.as<Variable>();
    if (varA == nullptr || varB == nullptr) {
      return false;
    }

    for (auto kv : lvMap_) {
      if (kv.second.get() == varA && body == varB) {
        return true;
      }
    }

    return false;
  }

  std::unordered_map<std::string, VarExpr> lvMap_;
};

/**
 * Unify allocate memory
 * @param [in] stmt  The statment to be transformed
 * @return           Transformed stmt
 */
Stmt UnifyAllocate(const Stmt &stmt) { return AllocateUnify().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
