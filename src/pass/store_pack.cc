/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <tvm/ir_pass.h>
#include "emit_insn/insn_info.h"
#include "emit_insn/ir_transform.h"
#include "analyze_align.h"

namespace akg {
namespace ir {

class ReducePacker : public IRMutator {
 public:
  ReducePacker() = default;
  ~ReducePacker() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_ub_gm" || (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
                                           !exclude_align_analyze_list.count(op->value.as<StringImm>()->value))) {
      IRInfo info;
      ParserVisitor(info, false).Run(s);
      if (info.ChangeLastDimReduce()) {
        auto body = info.GenStmt();
        return AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(info.arith_info.insn_type), body);
      }
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }
};

Stmt PackStore(Stmt stmt) {
  stmt = TransposeTransform().Mutate(stmt);
  stmt = ReducePacker().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg