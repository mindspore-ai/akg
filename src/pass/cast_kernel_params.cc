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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include "pass/ir_util.h"

namespace akg {
namespace ir {
void ReplaceBinds(Array<NodeRef> &arg_list, const std::unordered_map<Var, Expr, NodeHash, NodeEqual> &var_map) {
  auto arg_list_size = arg_list.size();
  for (size_t arg_i = 0u; arg_i < arg_list_size; ++arg_i) {
    if (auto buf = arg_list[arg_i].as<BufferNode>()) {
      Array<Expr> new_shape = buf->shape;
      for (size_t i = 0u; i < buf->shape.size(); ++i) {
        new_shape.Set(i, Substitute(buf->shape[i], var_map));
      }
      Buffer new_buf = BufferNode::make(buf->data, buf->dtype, new_shape, buf->strides, buf->elem_offset, buf->name,
                                        buf->scope, buf->data_alignment, buf->offset_factor, buf->buffer_type);
      arg_list.Set(arg_i, new_buf);
    }
  }
}

class PeelSpecialAttrStmt : public IRMutator {
 public:
  explicit PeelSpecialAttrStmt(std::vector<Stmt> &outer_stmts) : outer_stmts_(outer_stmts) {}
  ~PeelSpecialAttrStmt() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "thread_extent") {
      outer_stmts_.emplace_back(AttrStmt::make(op->node, op->attr_key, op->value, Evaluate::make(0)));
      return op->body;  // assumes no nested thread_extent
    }
    return IRMutator::Mutate_(op, s);
  }

  std::vector<Stmt> &outer_stmts_;
};

Array<NodeRef> CastKernelParams(const Stmt &stmt, const Array<NodeRef> &arg_list) {
  std::vector<Stmt> let_stmts;
  auto body_stmt = PeelSpecialAttrStmt(let_stmts).Mutate(stmt);
  Array<NodeRef> new_arg_list;
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> var_map;
  for (const auto &arg : arg_list) {
    if (auto var = arg.as<Variable>()) {
      if (var->type.is_int() && var->type.bits() == 64) {
        new_arg_list.push_back(arg);
      } else {
        Var param_var = Variable::make(Int(64), var->name_hint + "_I64");
        let_stmts.emplace_back(
          LetStmt::make(ktvm::Downcast<Var>(arg), Cast::make(var->type, param_var), Evaluate::make(0)));
        new_arg_list.push_back(param_var);
        var_map[ktvm::Downcast<Var>(arg)] = Cast::make(var->type, param_var);
      }
    } else {
      new_arg_list.push_back(arg);
    }
  }
  Stmt new_stmt = ktvm::ir::MergeNest(let_stmts, body_stmt);
  ReplaceBinds(new_arg_list, var_map);

  Array<NodeRef> res;
  res.push_back(new_stmt);
  res.push_back(new_arg_list);
  return res;
}
}  // namespace ir
}  // namespace akg
