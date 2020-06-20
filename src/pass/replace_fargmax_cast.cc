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

/**
 *  ir_before:
 * // attr [0] pragma_emit_insn = "elewise_binary_fargmax"
 * for (cc3, 0, 3) {
 *   for (cc4, 0, 3) {
 *     for (cc5, 0, 1020) {
 *        compute_local_UB[((cc3*3) + cc4)] = fargmax(compute_local_UB[((cc3*3) + cc4)] if 9, data_local_UB[(((cc3*3060)
 + (cc4*1020)) + cc5)] if 1020) if 9
 *      }
 *    }
 *  }
 *   // attr [cast1_local_UB] storage_scope = "local.UB"
 *  allocate cast1_local_UB[int32 * 16]
 *  // attr [0] pragma_emit_insn = "elewise_single_Cast"
 * for (cc3, 0, 3) {
 *  for (cc4, 0, 3) {
 *   cast1_local_UB[((cc3*3) + cc4)] = int32(compute_local_UB[((cc3*3) + cc4)] if 9) if 9
 *  }
 * }
 *
 * ir_after:
 *  // attr [cast1_local_UB] storage_scope = "local.UB"
 *  allocate cast1_local_UB[int32 * 16]
 *  // attr [0] pragma_emit_insn = "vec_argmax_Cast"
 * for (cc3, 0, 3) {
 *  for (cc4, 0, 3) {
 *   cast1_local_UB[((cc3*3) + cc4)] = int32(compute_local_UB[((cc3*3) + cc4)] if 9) if 9
 *  }
 * }
*/

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include "tvm.h"

namespace akg {
namespace ir {
class ReplaceFargmaxCast : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn") {
      Stmt body = op->body;
      CHECK(op->value.as<StringImm>());
      std::string str = op->value.as<StringImm>()->value;
      std::set<std::string> argIntrins = {"vec_binary_fargmax", "vec_binary_fargmin"};
      const std::string castIntrinName = "vec_single_cast";

      // Remove for and if
      while (body.as<For>() || body.as<IfThenElse>()) {
        const auto f = body.as<For>();
        const auto g = body.as<IfThenElse>();
        if (f) {
          body = f->body;
        } else {
          body = g->then_case;
        }
      }

      if (argIntrins.count(str) != 0) {
        // get buffer_var for compare and set flag of argmax
        const auto store = body.as<Store>();
        CHECK(store);
        var_ = store->buffer_var;
        is_argmax = true;
      } else if (str == castIntrinName) {
        const auto store = body.as<Store>();
        CHECK(store);
        const auto cast = store->value.as<Cast>();
        CHECK(cast);
        const auto load = cast->value.as<Load>();
        // only compare buffer_var(index not needed right now)
        if (load && cast->type == Int(32) && Equal(load->buffer_var, var_) && is_argmax) {
          // replace label elewise_single_Cast by elewise_single_fargmax_Cast
          Stmt stmt = AttrStmt::make(VarExpr("0", Int(32)), "pragma_emit_insn", Expr("vec_argmax_cast"), op->body);
          is_argmax = false;
          return stmt;
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  VarExpr var_;
  bool is_argmax{false};
};

Stmt ReplaceFargmaxCasts(Stmt stmt) {
  stmt = ReplaceFargmaxCast().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
