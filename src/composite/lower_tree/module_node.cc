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

#include "composite/lower_tree/module_node.h"
#include "build_module.h"
#include "codegen/lower.h"
#include "codegen/stage_lower.h"

namespace akg {
namespace lower {
void ModuleLowerNode::Process() {
  CHECK(children_.size() == 1);
  children_[0]->Run(this);
  auto build_rst = BuildRstNode::make(children_[0]->Node(), children_[0]->Data()->name);
  CHECK(build_rst.defined());
  module_ = BuildToModule(build_rst, children_[0]->Data()->target);
}

Array<NodeRef> ModuleLowerNode::GetArgs() {
  return Array<NodeRef>({children_[0]->Node(), children_[0]->Data()->arg_list_0});
}

BaseLowerNodePtr CreateModuleLowerNode(const std::string &, bool, const Map<std::string, NodeRef> &) {
  return std::make_shared<ModuleLowerNode>();
}

REG_NODE_CREATOR(kLlvm, kModule, CreateModuleLowerNode);
REG_NODE_CREATOR(kCuda, kModule, CreateModuleLowerNode);
REG_NODE_CREATOR(kCce, kModule, CreateModuleLowerNode);
}  // namespace lower
}  // namespace akg
