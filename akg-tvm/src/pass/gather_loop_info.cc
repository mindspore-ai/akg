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
#include "pass/utils.h"

namespace akg {
namespace ir {
class LoopInfoGather : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>()->value == "dma_copy") {
      inside_pragma = true;
      static_cast<void>(IRMutator::Mutate_(op, s));
      if (is_ub_to_gm) {
        // Do again to get for info
        static_cast<void>(IRMutator::Mutate_(op, s));
        Expr ext_comment;
        for (auto for_op : external_for_list) {
          Expr sub_comment = for_op->loop_var;
          sub_comment = Sub::make(sub_comment, for_op->extent);

          if (ext_comment.defined()) {
            ext_comment = Mul::make(ext_comment, sub_comment);
          } else {
            ext_comment = sub_comment;
          }
        }
        if (!ext_comment.defined()) {
          ext_comment = 0;
        }
        Expr int_comment;
        for (auto for_op : internal_for_list) {
          Expr sub_comment = for_op->loop_var;
          sub_comment = Sub::make(sub_comment, for_op->extent);

          if (int_comment.defined()) {
            int_comment = Mul::make(int_comment, sub_comment);
          } else {
            int_comment = sub_comment;
          }
        }
        if (!int_comment.defined()) {
          int_comment = 0;
        }

        auto stmt = AttrStmt::make(make_zero(Int(32)), "internal_for_loop", int_comment, s);
        stmt = AttrStmt::make(make_zero(Int(32)), "external_for_loop", ext_comment, stmt);
        stmt = AttrStmt::make(make_zero(Int(32)), "gm_addr", gm_addr, stmt);
        inside_pragma = false;
        internal_for_list.clear();
        is_ub_to_gm = false;
        return stmt;
      }
      // Detection of "write after read on the same buffer" is necessary
      if (is_gm_to_ub) {
        auto stmt = AttrStmt::make(make_zero(Int(32)), "gm_addr_rw", gm_addr, s);
        is_gm_to_ub = false;
        return stmt;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt stmt;
    if (is_ub_to_gm) {
      internal_for_list.push_back(op);
      stmt = IRMutator::Mutate_(op, s);
    } else {
      external_for_list.push_back(op);
      stmt = IRMutator::Mutate_(op, s);
      external_for_list.pop_back();
    }
    return stmt;
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (inside_pragma && GetBufScope(op->buffer_var->name_hint) == "global") {
      is_ub_to_gm = true;
      gm_addr = op->buffer_var;
    } else if (op->value->IsInstance<Load>()) {
      auto v = op->value.as<Load>();
      if (inside_pragma && GetBufScope(v->buffer_var->name_hint) == "global") {
        is_gm_to_ub = true;
        gm_addr = v->buffer_var;
      }
    }
    return s;
  }

 private:
  bool inside_pragma{false};
  bool is_ub_to_gm{false};
  bool is_gm_to_ub{false};
  Var gm_addr;
  std::vector<const For *> external_for_list;
  std::vector<const For *> internal_for_list;
};

Stmt GatherLoopInfo(Stmt stmt) {
  stmt = LoopInfoGather().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
