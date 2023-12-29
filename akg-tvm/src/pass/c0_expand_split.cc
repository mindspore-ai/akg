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

#include "tvm.h"

namespace akg {
namespace ir {
/*
 *  for (n, 0, 32) {
 *    for (c1, 0, 2) {
 *      for (h, 0, 33) {
 *        for (w, 0, 33) {
 *          for (c0, 0, 32) {
 *            output[n, c1, h, w, c0] = input[n, 2 * c1 + c0 / 16, h, w, c0 % 16]
 *          }
 *        }
 *      }
 *    }
 *  }
 *
 *  Transform to:
 *
 *  for (n, 0, 32) {
 *    for (c1, 0, 2) {
 *      for (h, 0, 33) {
 *        for (w, 0, 33) {
 *          for (c0, 0, 16) {
 *            output[n, c1, h, w, c0] = input[n, 2 * c1, h, w, c0]
 *          }
 *        }
 *      }
 *    }
 *  }
 *  for (n, 0, 32) {
 *    for (c1, 0, 2) {
 *      for (h, 0, 33) {
 *        for (w, 0, 33) {
 *          for (c0, 0, 16) {
 *            output[n, c1, h, w, c0 + 16] = input[n, 2 * c1 + 1, h, w, c0]
 *          }
 *        }
 *      }
 *    }
 *  }
 *
 */

#define STATUS_CLEAR (0)
#define DIV_MATCH (1)
#define MOD_MATCH (1 << 1)
#define FOR_EXPAND (1 << 8)
#define BLOCK_SIZE (16)
#define DOUBLE_BLOCK_SIZE (16 * 2)

const int TENSOR_DIM_FIVE = 5;
const int TENSOR_INDEX_ZERO = 0;
const int TENSOR_INDEX_ONE = 1;
const int TENSOR_INDEX_TWO = 2;
const int TENSOR_INDEX_THREE = 3;
const int TENSOR_INDEX_FOUR = 4;

enum RepeatType { Init = 0, First, Second };

class ExpandC0Split : public IRMutator {
 public:
  ExpandC0Split() {}
  ~ExpandC0Split() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn") {
      Stmt stmt = IRMutator::Mutate_(op, s);

      if ((status_ & (DIV_MATCH | MOD_MATCH)) == (DIV_MATCH | MOD_MATCH)) {
        repeat_ = First;
        Stmt stmt1 = IRMutator::Mutate_(op, s);
        repeat_ = Second;
        Stmt stmt2 = IRMutator::Mutate_(op, s);
        repeat_ = Init;

        return Block::make(stmt1, stmt2);
      }

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    loopvarMap_.emplace(std::pair<std::string, VarExpr>{name, var});
    Stmt stmt = IRMutator::Mutate_(op, s);
    loopvarMap_.erase(name);

    if (status_ != (DIV_MATCH | MOD_MATCH)) {
      return stmt;
    }

    const For *node = stmt.as<For>();
    CHECK(node);
    CHECK(is_const_int(node->extent, DOUBLE_BLOCK_SIZE) && is_const_int(node->min, 0));
    status_ |= FOR_EXPAND;

    return For::make(node->loop_var, 0, BLOCK_SIZE, node->for_type, node->device_api, node->body);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    const Call *call = op->value.as<Call>();
    if (call && (call->args.size() == TENSOR_DIM_FIVE) && (op->args.size() == TENSOR_DIM_FIVE)) {
      status_ = STATUS_CLEAR;

      div_ = true;
      static_cast<void>(this->Mutate(call->args[TENSOR_INDEX_ONE]));
      div_ = false;

      mod_ = true;
      static_cast<void>(this->Mutate(call->args[TENSOR_INDEX_FOUR]));
      mod_ = false;

      if (status_ != (DIV_MATCH | MOD_MATCH)) {
        return IRMutator::Mutate_(op, s);
      }

      if (repeat_ == First) {
        Expr body = this->Mutate(op->value);
        return Provide::make(op->func, op->value_index, body, op->args);
      }

      if (repeat_ == Second) {
        Expr body = this->Mutate(op->value);
        return Provide::make(op->func, op->value_index, body,
                             {op->args[TENSOR_INDEX_ZERO], op->args[TENSOR_INDEX_ONE], op->args[TENSOR_INDEX_TWO],
                              op->args[TENSOR_INDEX_THREE], op->args[TENSOR_INDEX_FOUR] + BLOCK_SIZE});
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Div *op, const Expr &e) final {
    if (div_ && is_const_int(op->b, BLOCK_SIZE)) {
      status_ |= DIV_MATCH;
    }

    if (repeat_ == First) {
      return Expr(0);
    }
    if (repeat_ == Second) {
      return Expr(1);
    }

    return e;
  }

  Expr Mutate_(const Mod *op, const Expr &e) final {
    if (mod_ && is_const_int(op->b, BLOCK_SIZE)) {
      status_ |= MOD_MATCH;
    }

    if (repeat_ == First || repeat_ == Second) {
      return op->a;
    }

    return e;
  }

 private:
  std::unordered_map<std::string, VarExpr> loopvarMap_;
  bool div_{false};
  bool mod_{false};
  RepeatType repeat_{Init};
  unsigned int status_{STATUS_CLEAR};
};

Stmt ExpandC0(Stmt stmt) { return ExpandC0Split().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
