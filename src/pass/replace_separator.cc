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
#include <tvm/expr.h>
#include <ir_pass.h>

namespace akg {
namespace ir {
class ReplaceSeparatorMutator : public IRMutator {
 public:
  ReplaceSeparatorMutator() {}
  ~ReplaceSeparatorMutator() override = default;
        
 private:
  std::string ReplaceString(std::string name_hint) {
    char old_value = '.';
    char new_value = '_';
    std::string new_var = name_hint;
    std::replace(new_var.begin(), new_var.end(), old_value, new_value);
    return new_var;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    Stmt body = this->Mutate(op->body);
    if (op->attr_key == air::ir::attr::reduce_update) {
      Array<IterVar> iter_vars = Downcast<Array<IterVar>>(op->node);
      Array<IterVar> new_iter_vars = MutateAxis(iter_vars);
      return AttrStmt::make(new_iter_vars, op->attr_key, op->value, body);
    }
    if (auto compute = op->node.as<ComputeOpNode>()) {
      if (ReplaceString(compute->name) != compute->name) {
        return AttrStmt::make(MutateComputeOp(compute), op->attr_key, op->value, body);
      }
    }
    return AttrStmt::make(op->node, op->attr_key, op->value, body);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    Stmt body = this->Mutate(op->body);
    if (const auto compute = op->func.as<ComputeOpNode>()) {
      if (ReplaceString(compute->name) != compute->name) {
        return Realize::make(MutateComputeOp(compute), op->value_index, op->type, op->bounds, op->condition, body);
      }
    }
    return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) override {
    Stmt body = this->Mutate(op->body);
    if (const auto compute = op->func.as<ComputeOpNode>()) {
      if (ReplaceString(compute->name) != compute->name) {
        return ProducerConsumer::make(MutateComputeOp(compute), op->is_producer, body);
      }
    }
    return ProducerConsumer::make(op->func, op->is_producer, body);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    auto stmt = IRMutator::Mutate_(op, s);
    auto new_op = stmt.as<Provide>();
    CHECK(new_op);
    if (const auto compute = new_op->func.as<ComputeOpNode>()) {
      if (ReplaceString(compute->name) != compute->name) {
        return Provide::make(MutateComputeOp(compute), new_op->value_index, new_op->value, new_op->args);
      }
    }
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    auto expr = IRMutator::Mutate_(op, e);
    auto new_op = expr.as<Call>();
    CHECK(new_op);
    if (const auto compute = new_op->func.as<ComputeOpNode>()) {
      std::string name = ReplaceString(compute->name);
      if (name != compute->name) {
        Operation func = MutateComputeOp(compute);
        return Call::make(new_op->type, name, new_op->args, new_op->call_type, func, new_op->value_index);
      }
    }
    return expr;
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    Stmt body = this->Mutate(op->body);
    std::string name = ReplaceString(op->loop_var->name_hint);
    if (name == op->loop_var->name_hint) {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }
    if (var_map.count(op->loop_var.get()) == 0) {
      Var replaced_loop_var = Variable::make(op->loop_var.type(), name);
      var_map[op->loop_var.get()] = replaced_loop_var;
    }
    return For::make(var_map[op->loop_var.get()], op->min, op->extent, op->for_type, op->device_api, body);
  }

  Expr Mutate_(const Variable *op, const Expr &e) override {
    std::string name = ReplaceString(op->name_hint);
    if (name == op->name_hint) {
      return e;
    }
    if (var_map.count(op) == 0) {
      Var replaced_var = Variable::make(e.type(), name);
      var_map[op] = replaced_var;
    }
    return var_map[op];
  }

  Array<IterVar> MutateAxis(const Array<IterVar> axis) {
    Array<IterVar> new_axis;
    for (IterVar iv : axis) {
      if (var_map.count(iv->var.get()) == 0) {
        std::string name = ReplaceString(iv->var->name_hint);
        Var loop_var = Variable::make(iv->var.type(), name);
        var_map[iv->var.get()] = loop_var;
      }
      IterVar replaced_iv = IterVarNode::make(iv->dom, var_map[iv->var.get()], iv->iter_type, iv->thread_tag);
      new_axis.push_back(replaced_iv);
    }
    return new_axis;
  }

  Operation MutateComputeOp(const ComputeOpNode *compute) {
    if (op_map.count(compute) == 0) {
      std::string name = ReplaceString(compute->name);
      Array<IterVar> axis = MutateAxis(compute->axis);
      Array<Expr> body;
      for (const Expr e : compute->body) {
        const auto reduce = e.as<Reduce>();
        if (reduce) body.push_back(e);
        else body.push_back(this->Mutate(e));
      }
      op_map[compute] = ComputeOpNode::make(name, compute->tag, compute->attrs, axis, body);
    }
    return op_map[compute];
  }

  std::unordered_map<const Variable *, Var> var_map;
  std::unordered_map<const ComputeOpNode *, Operation > op_map;
};

Stmt ReplaceSeparator(Stmt stmt) {
  stmt = ReplaceSeparatorMutator().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace akg