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

#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>

#include <unordered_map>

#include "common/common_util.h"
#include "pass/utils.h"
#include "ir_pass.h"

namespace akg {
namespace ir {
constexpr auto REPLACE_VAR = "REPLACE_VAR";

class InsertIfForParallel : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->for_type != ForType::Parallel) {
      first_loop_var_ = 0;
      return IRMutator::Mutate_(op, s);
    }

    if (Equal(first_loop_var_, 0)) {
      first_loop_var_ = op->loop_var;
    }

    if (op->body.as<Block>() == nullptr) {
      return IRMutator::Mutate_(op, s);
    }

    auto block = op->body.as<Block>();
    auto first = block->first;
    auto rest = block->rest;

    if (first.as<For>() == nullptr || rest.as<For>() == nullptr) {
      return IRMutator::Mutate_(op, s);
    }

    auto first_for = first.as<For>();
    auto rest_for = rest.as<For>();
    if (first_for->for_type != ForType::Parallel || rest_for->for_type != ForType::Serial) {
      return IRMutator::Mutate_(op, s);
    }

    Expr condition = Mod::make(first_loop_var_, op->extent);
    condition = EQ::make(condition, 0);

    Stmt if_stmt = IfThenElse::make(condition, rest);
    Stmt attrs_stmt = AttrStmt::make(Expr("INFO"), REPLACE_VAR, Expr(REPLACE_VAR), if_stmt);
    Stmt block_stmt = Block::make(first, attrs_stmt);
    first_loop_var_ = 0;
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, block_stmt);
  }

  Expr first_loop_var_{0};
};
class FuseParallelLoop : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->for_type == ForType::Parallel) {
      if (parallel_nums_ > 0) {
        value_map_[op->loop_var.get()] = op->extent;
        ++parallel_nums_;
        return IRMutator::Mutate(op->body);
      } else {
        ++parallel_nums_;
        Stmt body = IRMutator::Mutate(op->body);

        if (parallel_nums_ == 1) {
          parallel_nums_ = 0;
          value_map_.clear();
          return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, op->body);
        }

        // Calculate extent after merged parallel loop.
        int init_i = 0;
        Expr div_extend = 1;
        Expr extent_sum = 1;
        std::unordered_map<const Variable *, Expr> vmap;
        for (auto item : value_map_) {
          if (init_i == 0) {
            vmap[item.first] = Mod::make(op->loop_var, item.second);
            div_extend = Mul::make(div_extend, item.second);
          } else {
            auto tmp_div = Div::make(op->loop_var, div_extend);
            vmap[item.first] = Mod::make(tmp_div, item.second);
          }
          extent_sum = Mul::make(extent_sum, item.second);
        }
        vmap[op->loop_var.get()] = Div::make(op->loop_var, extent_sum);

        Stmt new_body = Substitute(body, vmap);
        parallel_nums_ = 0;
        value_map_.clear();
        return For::make(op->loop_var, op->min, Mul::make(extent_sum, op->extent), op->for_type, op->device_api,
                         new_body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == REPLACE_VAR) {
      if (auto if_op = op->body.as<IfThenElse>()) {
        replace_expr_.push_back(if_op->condition);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<const Variable *, Expr> value_map_;
  int parallel_nums_{0};
  int max_parallel_{0};

 public:
  std::vector<Expr> replace_expr_;
};

class ReplaceIfForParallel : public IRMutator {
 public:
  explicit ReplaceIfForParallel(std::vector<Expr> &replace_expr) : replace_expr_(replace_expr) {}

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == REPLACE_VAR) {
      if (auto if_op = op->body.as<IfThenElse>()) {
        CHECK(replace_expr_.size() > 0);

        Expr new_condition = replace_expr_.front();
        Stmt stmt = IfThenElse::make(new_condition, if_op->then_case, if_op->else_case);
        replace_expr_.erase(replace_expr_.begin());
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 public:
  std::vector<Expr> replace_expr_;
};

Stmt AdjustParallelLoop(const Stmt &stmt) {
  InsertIfForParallel insert_if;
  Stmt s = insert_if.Mutate(stmt);

  FuseParallelLoop fuse_parallel;
  s = fuse_parallel.Mutate(s);

  ReplaceIfForParallel replace_parallel(fuse_parallel.replace_expr_);
  return replace_parallel.Mutate(s);
}
}  // namespace ir
}  // namespace akg
