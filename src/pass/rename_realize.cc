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
#include <ir_pass.h>
#include <tvm/operation.h>
#include <pass/utils.h>

namespace akg {
namespace ir {
class RealizeRenamer : public IRMutator {
 public:
  explicit RealizeRenamer(const Map<Tensor, Buffer> &extern_buffer) {
    for (auto kv : extern_buffer) {
      global_.insert({kv.first->op.get(), kv.first->op});
      CHECK_EQ(attr_name_.count(kv.first->op->name), 0) << "Duplicate name of global Tensor in binds ";
      attr_name_.insert(kv.first->op->name);
    }
  }

  ~RealizeRenamer() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    const auto cop = op->node.as<ComputeOpNode>();
    if (cop && op->attr_key == ktvm::ir::attr::realize_scope && global_.count(op->node.get()) == 0) {
      // don't touch empty scope
      const auto st = op->value.as<StringImm>();
      if (!st) {
        return IRMutator::Mutate_(op, s);
      }
      std::string name = cop->name;
      if (st->value == "local.UB" || st->value.empty()) {
        std::string extend;
        // extend "local.UB" to the attr with realize_scope "local.UB" but not has "local.UB" in its name
        if (st->value == "local.UB" && name.find("local.UB") == std::string::npos &&
            name.find("local_UB") == std::string::npos) {
          extend = ".local.UB";
        }
        // rename overlapping attr
        if (attr_name_.count(name + extend) == 0) {
          name = name + extend;
        } else {
          std::string str;
          do {
            str = "_rename" + std::to_string(++m);
          } while (attr_name_.count(name + str + extend) != 0);
          name = name + str + extend;
        }
        if (name != cop->name) {
          attr_name_.insert(name);
          auto n = ComputeOpNode::make(name, cop->tag, cop->attrs, cop->axis, cop->body);
          replace_[op->node.get()] = n;
          Stmt body = this->Mutate(op->body);
          return AttrStmt::make(n, op->attr_key, op->value, body);
        }
      }
      attr_name_.insert(name);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<Realize>();
    CHECK(n);
    if (replace_.count(op->func.get())) {
      stmt = Realize::make(replace_[op->func.get()], n->value_index, n->type, n->bounds, n->condition, n->body);
    }
    return stmt;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<ProducerConsumer>();
    CHECK(n);
    if (replace_.count(op->func.get())) {
      stmt = ProducerConsumer::make(replace_[op->func.get()], n->is_producer, n->body);
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<Provide>();
    CHECK(n);
    if (replace_.count(op->func.get())) {
      stmt = Provide::make(replace_[op->func.get()], n->value_index, n->value, n->args);
    }
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    const auto n = expr.as<Call>();
    CHECK(n);
    if (replace_.count(op->func.get())) {
      auto oper = replace_[op->func.get()];
      expr = Call::make(n->type, oper->name, n->args, n->call_type, oper, n->value_index);
    }
    return expr;
  }

 private:
  std::unordered_map<const Node *, ktvm::Operation> replace_;
  std::unordered_map<const Node *, ktvm::Operation> global_;
  std::set<std::string> attr_name_;
  int m{0};
};

/**
 * RenameRealize
 * @param [in] stmt             stmt to be renamed
 * @param [in] extern_buffer    global address
 * @param [in] replace          a map to modify stmt
 * @return                      renamed stmt
 */
Stmt RenameRealize(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, const Map<Tensor, Tensor> &replace) {
  for (const auto &it : replace) {
    stmt = TensorSubstitute(stmt, it.first->op, it.second->op, it.second->value_index);
  }
  stmt = RealizeRenamer(extern_buffer).Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
