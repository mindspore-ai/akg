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

#include <algorithm>
#include <set>
#include <vector>
#include <utility>
#include "composite/lower_tree/base_node.h"
#include "composite/lower_tree/json_leaf.h"

namespace akg {
namespace lower {
Schedule GetScheduleWithBuildInfo(const BuildInfo &info) {
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  return create_schedule(ops);
}

class LowerNodeDecorator : public LowerRunner {
 public:
  LowerNodeDecorator(BaseLowerNode *lower, const std::function<void(BaseLowerNode*, LowerRunner*, StageType)> &fn,
                     LowerRunner *next)
   : fn_(fn), next_(next), lower_(lower) {}
  ~LowerNodeDecorator() = default;
  void Lower(StageType s) override {
    fn_(lower_, next_, s);
  }
 protected:
  std::function<void(BaseLowerNode*, LowerRunner*, StageType)> fn_;
  LowerRunner *next_;
  BaseLowerNode *lower_;
};

void BaseLowerNode::Decorate(const std::function<void(BaseLowerNode*, LowerRunner*, StageType)> &fn) {
  std::unique_ptr<LowerRunner> dec = std::make_unique<LowerNodeDecorator>(this, fn, runner_);
  runner_ = dec.get();
  decorators_.emplace_back(std::move(dec));
}

void BaseLowerNode::VisitLeaf(const std::function<void(JsonLowerLeaf *)> &fn) {
  std::vector<BaseLowerNode *> stack;
  stack.push_back(this);
  while (!stack.empty()) {
    BaseLowerNode *a = stack.back();
    stack.pop_back();
    if (a->name_ == "JsonLowerLeaf") {
      fn(static_cast<JsonLowerLeaf*>(a));
    }
    for (auto &child : a->children_) {
      stack.push_back(child.get());
    }
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
