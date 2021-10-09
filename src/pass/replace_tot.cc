/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>

#include <unordered_map>

#include "common/common_util.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
namespace {
class CollectTensorForPromote : public IRVisitor {
 public:
  CollectTensorForPromote() {}
  ~CollectTensorForPromote() override = default;

  void Visit_(const Provide *op) {
    auto func = op->func;
    if (!func.defined() || func.as<OperationNode>() == nullptr) {
      IRVisitor::Visit_(op);
    }
    auto operation = func.as<OperationNode>();
    if (operation->tag != "tot") {
      IRVisitor::Visit_(op);
    }
    const auto &attrs = operation->attrs;
    if (attrs.count("tensor_of_tensor_pos")) {
      auto tot_pos = ir::GetInt32Const(Downcast<Expr>(attrs["tensor_of_tensor_pos"]));
      CHECK(-1 <= tot_pos && tot_pos < (int)op->value.as<Call>()->args.size());
      if (tot_pos == -1) {
        tensor_not_promote_.insert(func->func_name());
      } else {
        tensor_not_promote_.insert(op->value.as<Call>()->args[tot_pos].as<Call>()->name);
      }
    }
    if (attrs.count("first_index_pos")) {
      auto first_index = ir::GetInt32Const(Downcast<Expr>(attrs["first_index_pos"]));
      CHECK(0 <= first_index && first_index < (int)op->value.as<Call>()->args.size());
      inner_tensor_.insert(op->value.as<Call>()->args[first_index].as<Call>()->name);
    }
  }

  std::unordered_set<std::string> GetNotPromote() const { return tensor_not_promote_; }
  std::unordered_set<std::string> GetInner() const { return inner_tensor_; }

 private:
  std::unordered_set<std::string> tensor_not_promote_;
  std::unordered_set<std::string> inner_tensor_;
};

/*RewriteTensorIndex has marked all tensor with INNER_TENSOR
  1 Change INNER_TENSOR to TENSOR_NOT_PROMOTE for not_promote tensors
  2 Keep INNER_TENSOR for inner tensors
  3 Remove attr for other tensors
  4 Add mark TENSOR_OF_TENSOR if tot ops found
 */
class ResetAttrMark : public IRMutator {
 public:
  explicit ResetAttrMark(const std::unordered_set<std::string> &inner,
                         const std::unordered_set<std::string> &not_promote)
      : inner_(inner), not_promote_(not_promote) {}
  ~ResetAttrMark() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    Stmt ret = IRMutator::Mutate(op->body);
    if (op->node.as<StringImm>() && op->node.as<StringImm>()->value == "INFO" && op->attr_key == "INNER_TENSOR" &&
        not_promote_.count(op->value.as<StringImm>()->value)) {
      return AttrStmt::make(op->node, "TENSOR_NOT_PROMOTE", op->value, ret);
    } else if (op->node.as<StringImm>() && op->node.as<StringImm>()->value == "INFO" &&
               op->attr_key == "INNER_TENSOR" && inner_.count(op->value.as<StringImm>()->value)) {
      return AttrStmt::make(op->node, op->attr_key, op->value, ret);
    } else if (op->node.as<StringImm>() && op->node.as<StringImm>()->value == "INFO" &&
               op->attr_key == "INNER_TENSOR") {
      return ret;
    }
    return AttrStmt::make(op->node, op->attr_key, op->value, ret);
  }

  Stmt Remark(const Stmt &stmt) {
    auto ret = IRMutator::Mutate(stmt);
    if (inner_.empty() && not_promote_.empty()) {
      return ret;
    }
    return AttrStmt::make(Expr("INFO"), "TENSOR_OF_TENSOR", Expr("TENSOR_OF_TENSOR"), ret);
  }

 private:
  const std::unordered_set<std::string> &inner_;
  const std::unordered_set<std::string> &not_promote_;
};

/*
   output(ax0, ax1) = Gather(input_0(ax0, ax1), input_1(ax0))
   -------------->
   output(ax0, ax1) = tot_op_id(input_0(ax0, ax1), input_1(ax0))
*/
class ReplaceOpNames : public IRMutator {
 public:
  ReplaceOpNames() {}
  ~ReplaceOpNames() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto operation = op->func.as<OperationNode>();
    if (operation == nullptr || operation->tag != "tot") {
      return IRMutator::Mutate_(op, s);
    }
    auto name = "tot_op_" + std::to_string(id_++);
    auto call = op->value.as<Call>();
    auto new_value = Call::make(call->type, name, call->args, call->call_type, call->func, call->value_index);
    auto provide = Provide::make(op->func, 0, new_value, op->args);
    tot_attrs_.Set(name, op->func.as<OperationNode>()->attrs);
    return provide;
  }

  Map<std::string, Map<std::string, NodeRef>> GetRes() { return tot_attrs_; }

 private:
  int id_{0};
  Map<std::string, Map<std::string, NodeRef>> tot_attrs_;
};
}  // namespace

Array<NodeRef> ReplaceTot(const Stmt &stmt) {
  auto collector = CollectTensorForPromote();
  collector.Visit(stmt);
  auto inner = collector.GetInner();
  auto not_promote = collector.GetNotPromote();
  auto reset_attr = ResetAttrMark(inner, not_promote);
  auto s0 = reset_attr.Remark(stmt);
  auto inst = ReplaceOpNames();
  auto s1 = inst.Mutate(s0);
  return Array<NodeRef>{s1, inst.GetRes()};
}
}  // namespace ir
}  // namespace akg
