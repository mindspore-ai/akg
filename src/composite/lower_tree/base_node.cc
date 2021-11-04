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

#include "composite/lower_tree/base_node.h"
#include <algorithm>
#include <set>
#include <vector>
#include <utility>

namespace akg {
namespace lower {
Schedule GetScheduleWithBuildInfo(const BuildInfo &info) {
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  return create_schedule(ops);
}

Map<std::string, NodeRef> AddNamePosfix(const std::string &name, const Map<std::string, NodeRef> &cur_forward_info,
                                        size_t idx, bool is_child, Map<std::string, NodeRef> forward_infos) {
  std::string posfix;
  if (cur_forward_info.find(kKernelNamePosfix) != cur_forward_info.end()) {
    posfix = cur_forward_info[kKernelNamePosfix].as<StringImm>()->value + "_";
  }
  posfix += name + "_";
  if (is_child) {
    posfix += std::to_string(idx);
  }
  forward_infos.Set(kKernelNamePosfix, Expr(posfix));
  return forward_infos;
}

void BaseLowerNode::Excute(BaseLowerNodePtr child, const Map<std::string, NodeRef> &forward_infos, bool is_clean,
                           bool pass_out_backward_info) {
  if (child->current_stage_ == StageType::Unknown ||
      (data_ && StageTypeLT(target_, child->current_stage_, entrance_stage_))) {
    auto pass_forward_info = is_clean ? Map<std::string, NodeRef>{} : forward_infos_;
    for (auto iter : forward_infos) {
      pass_forward_info.Set(iter.first, iter.second);
    }

    child->ReceiveForwardInfos(pass_forward_info);
    child->CleanBackwardInfos();
    child->Run(entrance_stage_);

    if (pass_out_backward_info) {
      UpdateBackwardInfos(child->BackwardInfos());
    }
  }
}

bool BaseLowerNode::IsSkipped() {
  if (forward_infos_.find(kCatch) != forward_infos_.end()) {
    std::set<std::string> catch_run;
    if (forward_infos_[kCatch]->IsInstance<Array<Expr>::ContainerType>()) {
      auto pass_nodes = Downcast<Array<Expr>>(forward_infos_[kCatch]);
      for (auto p : pass_nodes) {
        CHECK(p->IsInstance<StringImm>());
        catch_run.insert(p.as<StringImm>()->value);
      }
    } else {
      CHECK(forward_infos_[kCatch]->IsInstance<StringImm>());
      catch_run.insert(forward_infos_[kCatch].as<StringImm>()->value);
    }
    return catch_run.count(name_) == 0;
  }
  return false;
}

void BaseLowerNode::UpdateBackwardInfos(const Map<std::string, NodeRef> &backward_infos) {
  for (auto iter : backward_infos) {
    backward_infos_.Set(iter.first, iter.second);
  }
}

LowerNodeCreatorManager &LowerNodeCreatorManager::Instance() {
  static LowerNodeCreatorManager instance;
  return instance;
}

void LowerNodeCreatorManager::RegisterNodeCreator(const std::string &target, const std::string &node_type,
                                                  LowerNodeCreateFunc func) {
  if (creator_funcs_.find(target) == creator_funcs_.end()) {
    creator_funcs_.insert({target, {}});
  }

  CHECK(creator_funcs_[target].find(node_type) == creator_funcs_[target].end())
    << "Lower node creator of " << node_type << " for " << target << " is all ready exist!";

  creator_funcs_[target].insert({node_type, func});
}

LowerNodeCreateFunc LowerNodeCreatorManager::GetLowerNodeCreator(const std::string &target,
                                                                 const std::string &node_type) {
  CHECK(creator_funcs_.find(target) != creator_funcs_.end()) << "Target " << target << " is not supported!";
  CHECK(creator_funcs_[target].find(node_type) != creator_funcs_[target].end())
    << node_type << " lower node creator for target " << target << " is not supported!";
  return creator_funcs_[target][node_type];
}
}  // namespace lower
}  // namespace akg
