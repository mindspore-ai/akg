/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "composite/lower_tree/tot_node.h"

#include "composite/lower_tree/json_leaf.h"

namespace akg {
namespace lower {
namespace {
Stmt TotReplace(Stmt &stmt, LowerData &data, Map<std::string, Map<std::string, NodeRef>> *tot_attr,
                const Array<Tensor> &noinline_indeed) {
  Array<NodeRef> res = ir::ReplaceTot(stmt);
  *tot_attr = Downcast<Map<std::string, Map<std::string, NodeRef>>>(res[1]);
  // remove noinline_indeed from args
  Array<NodeRef> filter_args;
  for (const auto &old_arg : data->arg_list_0) {
    auto iter = std::find_if(noinline_indeed.begin(), noinline_indeed.end(), [old_arg](const Tensor &noinline) {
      if (old_arg.as<BufferNode>() != nullptr) {
        return Downcast<Buffer>(old_arg)->name == noinline->op->name;
      } else if (old_arg.as<TensorNode>() != nullptr) {
        return Downcast<Tensor>(old_arg)->op->name == noinline->op->name;
      }
      return false;
    });
    if (iter == noinline_indeed.end()) {
      filter_args.push_back(old_arg);
    }
  }
  data->arg_list_0 = filter_args;
  PassMgr::SetArgs(data->arg_list_0);
  // remove no_inline from binds_0
  Map<Tensor, Buffer> filter_binds_0;
  for (const auto &old_kv : data->binds_0) {
    auto iter = std::find_if(noinline_indeed.begin(), noinline_indeed.end(),
                             [old_kv](const Tensor &noinline) { return noinline->op->name == old_kv.first->op->name; });
    if (iter == noinline_indeed.end()) {
      filter_binds_0.Set(old_kv.first, old_kv.second);
    }
  }
  data->binds_0 = filter_binds_0;

  return Downcast<Stmt>(res[0]);
}

Stmt TotRecover(Stmt &stmt, LowerData &data, Map<std::string, Map<std::string, NodeRef>> *tot_attr) {
  return ir::RecoverTot(stmt, *tot_attr, data->binds_0);
}

void ModifyBackwardTot(Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> &, BuildInfo &info,
                       Map<std::string, NodeRef> *backward_infos) {
  if (forward_infos.find(kTot) == forward_infos.end()) {
    return;
  }

  Array<Tensor> noinline_indeed = info.opt.noinline_indeed;
  backward_infos->Set(kNoInlineIndeed, noinline_indeed);
}

void AttachTotDecorator(BaseLowerNode *child, Map<std::string, NodeRef> &forward_infos,
                        Map<std::string, NodeRef> *backward_infos) {
  auto func = [&forward_infos, backward_infos](BaseLowerNode *node, LowerRunner *next, StageType s) {
    JsonLowerLeaf *leaf = static_cast<JsonLowerLeaf*>(node);
    next->Lower(s);
    ModifyBackwardTot(forward_infos, leaf->Attrs(), leaf->info_, backward_infos);
  };
  child->VisitLeaf([&func](JsonLowerLeaf *node) { node->Decorate(func); });
}
}  // namespace

void TotLowerNode::Lower(StageType stage) {
  CHECK(children_.size() == 1);

  Map<std::string, NodeRef> forward_infos;
  Map<std::string, NodeRef> backward_infos;
  forward_infos.Set(kTot, Expr(true));
  AttachTotDecorator(children_[0].get(),forward_infos, &backward_infos);
  children_[0]->Run(this);

  auto data = children_[0]->Data();
  auto dump_mng = DumpManager(data->name + "_" + kTot, data->config->dump_pass_ir);
  auto noinline_indeed = Downcast<Array<Tensor>>(backward_infos[kNoInlineIndeed]);
  Map<std::string, Map<std::string, NodeRef>> tot_attr;
  auto TotReplaceBind = [&dump_mng, &tot_attr, &noinline_indeed](NodeRef &node_ref, LowerData &data) -> NodeRef {
    Stmt stmt = Downcast<Stmt>(node_ref);
    TRANSFORM_AND_TRY_DUMP(dump_mng, stmt, TotReplace, stmt, data, &tot_attr, noinline_indeed);
    return stmt;
  };
  auto TotRecoverBind = [&dump_mng, &tot_attr](NodeRef &node_ref, LowerData &data) -> NodeRef {
    Stmt stmt = Downcast<Stmt>(node_ref);
    TRANSFORM_AND_TRY_DUMP(dump_mng, stmt, TotRecover, stmt, data, &tot_attr);
    return stmt;
  };

  StageLower stage_lower(data);
  stage_lower.RunTo(entrance_stage_)
    .ApplyMutator(TotReplaceBind)
    .RunTo(StageType::Poly)
    .ApplyMutator(TotRecoverBind)
    .RunTo(stage);

  node_ref_ = stage_lower.Node();
  data_ = stage_lower.Data();
  current_stage_ = stage;
}

BaseLowerNodePtr CreateTotLowerNode(const std::string &target, bool, const Map<std::string, NodeRef> &) {
  return std::make_shared<TotLowerNode>(target);
}

REG_NODE_CREATOR(kCuda, kTot, CreateTotLowerNode);
}  // namespace lower
}  // namespace akg
