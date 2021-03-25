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
#include <ir_pass.h>

namespace akg {
namespace ir {

class TensorAccessRewriter : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "buffer_bind_scope") {
      Array<NodeRef> bind_spec = Downcast<Array<NodeRef>>(op->node);
      Buffer buffer = Downcast<Buffer>(bind_spec[0]);
      Tensor tensor = Downcast<Tensor>(bind_spec[1]);
      tensors_.emplace(buffer->data.get(), tensor);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->name == "tensor_load") {
      auto it = tensors_.find(op->args[0].as<Variable>());
      CHECK(it != tensors_.end());
      Tensor t = it->second;
      Array<Expr> args;
      for (size_t i = 1; i < op->args.size(); ++i) {
        args.push_back(op->args[i]);
      }
      return Call::make(t->dtype, t->op->name, args, Call::CallType::Halide, t->op, t->value_index);
    } 
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) override {
    const Call * call = op->value.as<Call>();
    if (call != nullptr && call->name == "tensor_store") {
      Expr expr = IRMutator::Mutate(op->value);
      call = expr.as<Call>();
      auto it = tensors_.find(call->args[0].as<Variable>());
      CHECK(it != tensors_.end());
      Expr value = call->args[1];
      Array<Expr> args;
      for (size_t i = 2; i < call->args.size(); ++i) {
        args.push_back(call->args[i]);
      }
      return Provide::make(it->second->op, 0, value, args);
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<const Variable*, Tensor> tensors_;
};

Stmt TensorAccessRewrite(const Stmt stmt) {
  return TensorAccessRewriter().Mutate(stmt);
}

}  // namespace ir
}  // namespace akg
