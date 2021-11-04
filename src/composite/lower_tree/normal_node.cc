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

#include "composite/lower_tree/normal_node.h"

#include "codegen/lower.h"
#include "codegen/stage_lower.h"
#include "composite/utils/util.h"

namespace akg {
namespace lower {
void NormalLowerNode::ExcuteImpl(StageType stage) {
  CHECK(children_.size() == 1);
  Excute(children_[0]);
  StageLower stage_lower(children_[0]->Data());
  stage_lower.RunTo(stage);
  node_ref_ = stage_lower.Node();
  data_ = stage_lower.Data();
  current_stage_ = stage;
}

BaseLowerNodePtr CreateNormalLowerNode(const std::string &target, bool, const Map<std::string, NodeRef> &) {
  return std::make_shared<NormalLowerNode>(target);
}

REG_NODE_CREATOR(kLlvm, kNormal, CreateNormalLowerNode);
REG_NODE_CREATOR(kCuda, kNormal, CreateNormalLowerNode);
REG_NODE_CREATOR(kCce, kNormal, CreateNormalLowerNode);
}  // namespace lower
}  // namespace akg
