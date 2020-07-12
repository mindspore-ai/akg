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

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>
#include <pass/utils.h>
#include "build_module.h"

namespace akg {
namespace ir {
inline bool IsArgsSame(const Array<Expr> &args1, const Array<Expr> &args2, bool can_remove_broadcast = false) {
  if (args1.size() != args2.size()) {
    return false;
  }

  auto size = args1.size();
  for (size_t i = 0; i < size; i++) {
    if (args1[i].as<Variable>() && args2[i].as<Variable>()) {
      auto var1 = args1[i].as<Variable>();
      auto var2 = args2[i].as<Variable>();
      CHECK(var1);
      CHECK(var2);
      if (var1->name_hint != var2->name_hint) {
        return false;
      }
    } else if (args1[i].as<IntImm>() && args2[i].as<IntImm>()) {
      auto var1 = args1[i].as<IntImm>();
      auto var2 = args2[i].as<IntImm>();
      CHECK(var1);
      CHECK(var2);
      if (var1->value != var2->value) {
        return false;
      }
    } else if (can_remove_broadcast && args1[i].as<Variable>() && args2[i].as<IntImm>()) {
      // broadcast case
      return is_zero(args2[i]);
    } else {
      return false;
    }
  }

  return true;
}

inline bool IsAttrValueSame(const Expr &value1, const Expr &value2) {
  if (value1.as<StringImm>() && value2.as<StringImm>()) {
    auto v1 = value1.as<StringImm>();
    auto v2 = value2.as<StringImm>();
    CHECK(v1);
    CHECK(v2);
    return (v1->value == v2->value);
  }
  return false;
}

struct CopyInfo {
  FunctionRef func_ref_;
  std::unordered_set<size_t> broadcast_indexes_;
};

class DetectCanEliminatedCopy : public IRVisitor {
 public:
  explicit DetectCanEliminatedCopy(const Map<Tensor, Buffer> &extern_buffers) : extern_buffers_(extern_buffers) {}
  ~DetectCanEliminatedCopy() override = default;

  void Visit_(const AttrStmt *op) final {
    auto func = Downcast<FunctionRef>(op->node);
    attr_[func] = op->value;
    IRVisitor::Visit(op->body);
    attr_.erase(func);
  }

  void Visit_(const Realize *op) final {
    auto func = op->func;
    realize_[func] = op;
    IRVisitor::Visit(op->body);
    realize_.erase(func);
  }

  void Visit_(const ProducerConsumer *op) final {
    auto func = op->func;
    producers_.insert(func);
    IRVisitor::Visit(op->body);
    producers_.erase(func);
  }

  void Visit_(const Provide *op) final {
    std::vector<const Call *> src_call;
    auto GetSrcCall = [&, this](const NodeRef &op) {
      if (const auto call = op.as<Call>()) {
        src_call.emplace_back(call);
      }
    };
    air::ir::PostOrderVisit(op->value, GetSrcCall);

    for (size_t i = 0; i < src_call.size(); ++i) {
      auto func = src_call[i]->func;
      if (copy_stmts_.count(func) != 0) {
        auto substitute_func = copy_stmts_[func].func_ref_;
        if (realize_.count(substitute_func) == 0) {
          // Do not elimate the copy if out of realize scope.
          copy_stmts_.erase(func);
          not_copy_.insert(func);
        }
      }
    }

    auto call_op = op->value.as<Call>();
    if (call_op == nullptr) {
      if (enable_compute_in_place_ && (op->value.as<Add>() || op->value.as<Sub>() || op->value.as<Mul>())) {
        // A(i0, i1, i2) = B(i0, i1, i2) * C(i0, i1), here A can be replaced by B to remove redundant buffer
        // if B is not used later.
        std::vector<const Call *> target_call;
        for (const auto &ele : src_call) {
          if (ele->func != op->func && IsArgsSame(op->args, ele->args, false)) {
            target_call.emplace_back(ele);
          }
        }
        if (target_call.size() == 1) {
          FunctionRef f;
          for (const auto &it : copy_stmts_) {
            if (it.second.func_ref_ == target_call[0]->func) {
              f = it.first;
            }
          }
          if (!f.defined()) {
            call_op = target_call[0];
          } else {
            // Do not eliminate the copy if detect B used again.
            copy_stmts_.erase(f);
            not_copy_.insert(f);
          }
        }
      }

      if (call_op == nullptr) {
        if (copy_stmts_.count(op->func) != 0) {
          copy_stmts_.erase(op->func);
        }
        not_copy_.insert(op->func);
        return;
      }
    }

    auto dst_args = op->args;
    auto src_args = call_op->args;
    auto op_func = op->func;
    auto call_op_func = call_op->func;

    if (!copy_stmts_.empty()) {
      for (auto &it : copy_stmts_) {
        if (op_func == it.first) {
          if (call_op_func != it.second.func_ref_) {
            // Do not eliminate the copy if detect any modification to the target buffer after the copy.
            copy_stmts_.erase(op_func);
            not_copy_.insert(op_func);
            return;
          }
        }
      }
    }

    // detect copy, eg. compute_1(i0, i1) = compute_2(i0, i1), in which compute_1 is not bound
    // only detect complete copy: the number of args must be the same with the size of bounds in realize
    if (not_copy_.count(op_func) == 0 && copy_stmts_.count(call_op_func) == 0 &&
        IsArgsSame(dst_args, src_args, can_remove_broadcast_) && realize_.count(op_func) != 0 &&
        realize_.count(call_op_func) != 0 && attr_.count(op_func) != 0 && attr_.count(call_op_func) != 0 &&
        IsAttrValueSame(attr_[op_func], attr_[call_op_func])) {
      if (dst_args.size() == realize_[op_func]->bounds.size()) {
        if (std::any_of(extern_buffers_.begin(), extern_buffers_.end(), [=](const std::pair<Tensor, Buffer> &it) {
              return (op_func->func_name() == it.first->op->name);
            })) {
          return;
        }
      }

      std::unordered_set<size_t> broadcast_indexes;
      if (can_remove_broadcast_) {
        for (size_t i = 0; i < src_args.size(); ++i) {
          if (is_zero(src_args[i])) {
            broadcast_indexes.insert(i);
          }
        }
      }
      copy_stmts_[op_func] = CopyInfo{call_op_func, broadcast_indexes};
    } else {
      // Do not eliminate the copy.
      if (copy_stmts_.count(op_func) != 0) {
        copy_stmts_.erase(op_func);
      }
      not_copy_.insert(op_func);
    }
  }

  std::unordered_map<FunctionRef /*copy_dst*/, CopyInfo /*copy_src*/, NodeHash, NodeEqual> copy_stmts_;

 private:
  std::unordered_map<FunctionRef, Expr, NodeHash, NodeEqual> attr_;
  std::unordered_map<FunctionRef, const Realize *, NodeHash, NodeEqual> realize_;
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> producers_;
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> not_copy_;
  const Map<Tensor, Buffer> &extern_buffers_;
  bool can_remove_broadcast_ = global_attrs.GetBoolAttr(kEnableRemoveBroadcastCopy, false);
  bool enable_compute_in_place_ = global_attrs.GetBoolAttr(kEnableComputeInPlace, false);
};

class EliminateCopyAndRealize : public IRMutator {
 public:
  explicit EliminateCopyAndRealize(const std::unordered_map<FunctionRef, CopyInfo, NodeHash, NodeEqual> &copy_stmts)
      : copy_stmts_(copy_stmts) {}
  ~EliminateCopyAndRealize() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    auto body = this->Mutate(op->body);

    if (op->attr_key == air::ir::attr::realize_scope) {
      auto node = op->node.as<OperationNode>();
      if (node) {
        for (auto &it : copy_stmts_) {
          if (node->name == it.first->func_name() && op->node.get() == it.first.get()) {
            return body;
          }
        }
      }
    }

    return AttrStmt::make(op->node, op->attr_key, op->value, body);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (op->is_producer && copy_stmts_.count(op->func) != 0) {
      return this->Mutate(op->body);
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (copy_stmts_.count(op->func) == 0) {
      return IRMutator::Mutate_(op, s);
    }

    return this->Mutate(op->body);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (copy_stmts_.count(op->func) != 0) {
      if (op->value.as<Call>() == nullptr) {
        return Provide::make(copy_stmts_[op->func].func_ref_, op->value_index, op->value, op->args);
      }
      return Evaluate::make(0);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &s) final {
    auto it = copy_stmts_.find(op->func);
    if (it != copy_stmts_.end()) {
      auto new_func_ref = it->second.func_ref_;
      Array<Expr> new_args;
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (it->second.broadcast_indexes_.count(i) == 0) {
          new_args.push_back(op->args[i]);
        } else {
          new_args.push_back(make_zero(op->args[i].type()));
        }
      }
      auto new_call =
        Call::make(op->type, new_func_ref->func_name(), new_args, op->call_type, new_func_ref, op->value_index);
      return new_call;
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  // copy_stmts_ is of std::unordered_map<copy_dst, copy_src, NodeHash, NodeEqual>
  std::unordered_map<FunctionRef, CopyInfo, NodeHash, NodeEqual> copy_stmts_;
};

/* Eliminate useless copy, e.g.
 *  x = y
 *  z = x * w + b
 * In this case 'x = y' will be eliminated and x will be replaced by y
 */
Stmt CopyPropagation(const Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  DetectCanEliminatedCopy detect_visitor(extern_buffer);
  detect_visitor.Visit(stmt);

  EliminateCopyAndRealize eliminator(detect_visitor.copy_stmts_);
  return RemoveNoOp(eliminator.Mutate(stmt));
}
}  // namespace ir
}  // namespace akg
