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
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <pass/ir_util.h>
#include <unordered_map>
#include <iostream>
#include <set>
#include <map>

namespace akg {
namespace ir {
using ValueIndex = int;
using TensorName = std::string;
using ValueIndexMapping = std::unordered_map<ValueIndex, FunctionRef>;
using TensorMapping = std::unordered_map<TensorName, ValueIndexMapping>;

class RewriteMultiValueFuncMutator : public IRMutator {
 public:
  explicit RewriteMultiValueFuncMutator(const Map<Tensor, Tensor> &multi_output_mapping) {
    for (auto item : multi_output_mapping) {
      Tensor old_tensor = item.first;
      Tensor new_tensor = item.second;
      func_ref_mapping[old_tensor->op->func_name()][old_tensor->value_index] = new_tensor->op;
    }
  }

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    // we only process AttrStmt of realize_scope because we don't know the data structure of other AttrStmts
    if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Stmt body = this->Mutate(op->body);
      Array<NodeRef> tuple = Downcast<Array<NodeRef>>(op->node);
      Tensor tensor = Downcast<Tensor>(tuple[1]);
      auto buffer = Downcast<Buffer>(tuple[0]);
      if (func_ref_mapping.count(tensor->op->func_name())) {
        Tensor new_tensor =
          Downcast<Operation>(get_function_ref(tensor->op->func_name(), tensor->value_index)).output(0);
        return AttrStmt::make(Array<NodeRef>{tuple[0], new_tensor}, op->attr_key, op->value, body);
      } else {
        return AttrStmt::make(tuple, op->attr_key, op->value, body);
      }
    }

    if (op->attr_key != air::ir::attr::realize_scope) {
      return IRMutator::Mutate_(op, s);
    }
    auto func_ref = Downcast<FunctionRef>(op->node);
    if (func_ref->num_outputs() == 1) {
      return IRMutator::Mutate_(op, s);
    }

    TensorName name = func_ref->func_name();
    attr_stmt_to_add[name].push_back(op);
    return IRMutator::Mutate(op->body);
  }

  void check_name(const TensorName &name) {
    if (rewrite_names.count(name) > 0) {
      LOG(FATAL) << "RewriteMultiValueFunc: try to create tensor " << name
                 << " but it is already defined, please modify DSL to avoid this name.";
    }
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    if (op->func->num_outputs() == 1) {
      return IRMutator::Mutate_(op, s);
    }

    TensorName name = op->func->func_name();
    check_name(name);
    if (!func_ref_mapping.count(name) || !func_ref_mapping[name].count(op->value_index)) {
      std::string new_name = op->func->func_name() + "_v" + std::to_string(op->value_index);
      rewrite_names.insert(new_name);
      Array<Expr> realize_region;
      for (auto bound : op->bounds) {
        realize_region.push_back(bound->extent);
      }
      auto new_funcref = PlaceholderOpNode::make(new_name, realize_region, op->type);
      func_ref_mapping[name][op->value_index] = new_funcref;
    }
    FunctionRef new_funcref = func_ref_mapping[name][op->value_index];

    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    CHECK(op != nullptr);
    Stmt realize_stmt = Realize::make(new_funcref, 0, op->type, op->bounds, op->condition, op->body);
    Stmt attr_stmt = realize_stmt;

    if (attr_stmt_to_add.count(name) > 0) {
      for (auto attr : attr_stmt_to_add[name]) {
        attr_stmt = AttrStmt::make(new_funcref, attr->attr_key, attr->value, attr_stmt);
      }
    }
    return attr_stmt;
  }

  FunctionRef get_function_ref(const TensorName &name, int value_index) {
    CHECK(func_ref_mapping.count(name) > 0) << "tensor " << name << " not defined";
    CHECK(func_ref_mapping[name].count(value_index) > 0)
      << "tensor " << name << " does not have value index " << value_index;
    return func_ref_mapping[name][value_index];
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    check_name(op->func->func_name());
    if (op->func->num_outputs() == 1) {
      return IRMutator::Mutate_(op, s);
    }
    FunctionRef new_funcref = get_function_ref(op->func->func_name(), op->value_index);
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    CHECK(op != nullptr);
    return Provide::make(new_funcref, 0, op->value, op->args);
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->call_type != Call::CallType::Halide) {
      return IRMutator::Mutate_(op, e);
    }
    check_name(op->func->func_name());
    if (op->func->num_outputs() == 1) {
      return IRMutator::Mutate_(op, e);
    }
    FunctionRef new_funcref = get_function_ref(op->func->func_name(), op->value_index);
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Call>();
    CHECK(op != nullptr);
    return Call::make(op->type, new_funcref->func_name(), op->args, op->call_type, new_funcref, 0);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (op->func.defined() && func_ref_mapping.count(op->func->func_name())) {
      auto body = this->Mutate(op->body);
      for (auto sub_values : func_ref_mapping[op->func->func_name()]) {
        body = ProducerConsumer::make(sub_values.second, op->is_producer, body);
      }
      return body;
    }
    return IRMutator::Mutate_(op, s);
  }

  TensorMapping func_ref_mapping;
  std::unordered_set<TensorName> rewrite_names;
  std::unordered_map<TensorName, std::vector<const AttrStmt *>> attr_stmt_to_add;
};

Stmt RewriteMultiValueFunc(Stmt stmt, const Map<Tensor, Tensor> &multi_output_mapping) {
  stmt = RewriteMultiValueFuncMutator(multi_output_mapping).Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
