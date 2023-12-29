/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "composite/lower_tree/elemany_node.h"

#include <sstream>
#include <vector>
#include "codegen/lower.h"
#include "codegen/stage_lower.h"
#include "composite/utils/dump.h"
#include "composite/lower_tree/json_leaf.h"

namespace akg {
namespace lower {
namespace {
/*
if(x(arg0,arg1,...argn)>0){
  elemany[0]=1;
}
===to
if(x(arg0,arg1,..argn)>0){
  elemany[arg0,arg1,..argn]=1;
}
*/
class ElemAnyToElemwise : public IRMutator {
 public:
  explicit ElemAnyToElemwise(const std::string &s) : output_name_(s) {}
  ~ElemAnyToElemwise() override = default;
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (!op->then_case.defined()) return IRMutator::Mutate_(op, s);
    auto then_stmt = op->then_case.as<Provide>();
    if (then_stmt && then_stmt->func->func_name() == output_name_) {
      Array<Expr> left_args;
      if (auto lt = op->condition.as<LT>()) {
        auto call = lt->b.as<Call>();
        if (call && Equal(lt->a, make_const(Bool(), 0))) {
          left_args = call->args;
        }
      } else if (auto gt = op->condition.as<GT>()) {
        auto call = gt->a.as<Call>();
        if (call && Equal(gt->b, make_const(Bool(), 0))) {
          left_args = call->args;
        }
      }
      if (!left_args.empty()) {
        auto new_then_stmt = Provide::make(then_stmt->func, then_stmt->value_index, then_stmt->value, left_args);
        return IfThenElse::make(op->condition, new_then_stmt, op->else_case);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  std::string output_name_;
};

/*
elemany[arg0,arg1,...argn]=1;
===to
elemany[0]=1;
*/
class RecoverElemAny : public IRMutator {
 public:
  explicit RecoverElemAny(const std::string &s) : output_name_(s) {}
  ~RecoverElemAny() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func->func_name() == output_name_) {
      return Provide::make(op->func, op->value_index, op->value, {Expr(0)});
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::string output_name_;
};
}  // namespace

void ElemAnyLowerNode::Lower(StageType stage) {
  CHECK(children_.size() == 1);
  children_[0]->Run(this);
  auto data = children_[0]->Data();

  auto dump_mng = DumpManager(data->name + "_" + kElemAny, data->config->dump_pass_ir);
  auto ElemAnytReplaceBind = [&dump_mng](NodeRef &node_ref, LowerData &data) -> NodeRef {
    Stmt stmt = Downcast<Stmt>(node_ref);
    stmt = ElemAnyToElemwise("elemany").Mutate(stmt);
    dump_mng.DumpStmt("ElemAnyToElemwise", stmt);
    return stmt;
  };
  auto ElemAnyRecoverBind = [&dump_mng](NodeRef &node_ref, LowerData &data) -> NodeRef {
    Stmt stmt = Downcast<Stmt>(node_ref);
    stmt = RecoverElemAny("elemany").Mutate(stmt);
    dump_mng.DumpStmt("RecoverElemAny", stmt);
    return stmt;
  };

  StageLower stage_lower(data);
  stage_lower.RunTo(entrance_stage_)
    .ApplyMutator(ElemAnytReplaceBind)
    .RunTo(StageType::Poly)
    .ApplyMutator(ElemAnyRecoverBind)
    .RunTo(stage);

  node_ref_ = stage_lower.Node();
  data_ = stage_lower.Data();
  current_stage_ = stage;
}

BaseLowerNodePtr CreateElemAnyLowerNode(const std::string &target, bool, const Map<std::string, NodeRef> &) {
  return std::make_shared<ElemAnyLowerNode>(target);
}

REG_NODE_CREATOR(kCuda, kElemAny, CreateElemAnyLowerNode);
}  // namespace lower
}  // namespace akg
