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
class FuseParallelLoop : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->for_type == ForType::Parallel) {
      if (has_parallel_) {
        var_extents_.Set(op->loop_var, op->extent);
        return IRMutator::Mutate(op->body);
      } else {
        has_parallel_ = true;
        Map<Var, Expr> new_vmap;
        std::swap(new_vmap, var_extents_);
        Stmt body = IRMutator::Mutate(op->body);

        Map<Var, Expr> vmap;
        Expr new_extent = 1;
        for (auto item : var_extents_) {
          new_extent = Mul::make(new_extent, item.second);
          vmap.Set(item.first, Mod::make(op->loop_var, item.second));
        }

        vmap.Set(op->loop_var, Div::make(op->loop_var, new_extent));

        Stmt new_body = Substitute(body, vmap);
        std::swap(new_vmap, var_extents_);
        has_parallel_ = false;
        return For::make(op->loop_var, op->min, Mul::make(new_extent, op->extent), op->for_type, op->device_api,
                         new_body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Var loop_var_;
  Map<Var, Expr> var_extents_;
  bool has_parallel_{false};
};

Stmt AdjustParallelLoop(const Stmt &stmt) {
  FuseParallelLoop fuse_parallel;
  return fuse_parallel.Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
