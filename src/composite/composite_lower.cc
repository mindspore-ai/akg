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

#include <stack>
#include "build_module.h"
#include "composite/lower_tree/json_leaf.h"
#include "composite/lower_tree/tune_node.h"
#include "composite/lower_tree/module_node.h"

namespace akg {
namespace lower {
namespace {
bool IsJsonLowerType(const std::string &type) {
  return !type.empty() && type[0] == 'P' && type.find_first_not_of("0123456789", 1) == std::string::npos;
}
}  // namespace

BaseLowerNodePtr GetCompositeLower(const std::string &target, const std::string &type_str, bool poly,
                                   const Map<std::string, NodeRef> &segment_infos) {
  auto type_idx = 0;
  auto id_pos = type_str.find_first_of("0123456789");
  if (id_pos != std::string::npos) {
    type_idx = std::stoi(type_str.substr(id_pos));
  }
  auto type = type_str.substr(0, id_pos);
  Map<std::string, NodeRef> type_construct_infos;
  if (segment_infos.find(type) != segment_infos.end()) {
    auto type_construct_infos_array = Downcast<Array<NodeRef>>(segment_infos[type]);
    type_construct_infos = Downcast<Map<std::string, NodeRef>>(type_construct_infos_array[type_idx]);
  }

  return LowerNodeCreatorManager::Instance().GetLowerNodeCreator(target, type)(target, poly, type_construct_infos);
}

BaseLowerNodePtr ConstructLowerTree(const std::string &target, bool poly, const std::string &build_str,
                                    const Map<std::string, NodeRef> &segment_infos) {
  /*
   * Example:
   *   build_str: "Module[Parallel[Stitch0[Normal[P0],Tot[P1]],Tot[P2],Normal[P3]]]"
   *   result:    ModuleNode ───> ParallelNode ─┬─> StitchNode ─┬─> NormalNode ───> JsonLeaf
   *                (root)                      │               └─> TotNode    ───> JsonLeaf
   *                                            ├─> TotNode    ───> JsonLeaf
   *                                            └─> NormalNode ───> JsonLeaf
   */
  std::stack<BaseLowerNodePtr> lower_stack;
  size_t cur_pos = 0;
  while (cur_pos < build_str.size()) {
    auto next_pos = build_str.find_first_of("[],", cur_pos);
    CHECK(next_pos != std::string::npos);
    auto type = build_str.substr(cur_pos, next_pos - cur_pos);
    switch (build_str[next_pos]) {
      case '[': {
        auto new_lower = GetCompositeLower(target, type, poly, segment_infos);
        lower_stack.push(new_lower);
      } break;
      case ',':
        break;
      case ']': {
        if (IsJsonLowerType(type)) {
          auto json_child = GetCompositeLower(target, type, poly, segment_infos);
          auto json_parent = lower_stack.top();
          json_parent->AddChild(json_child);
        }

        if (lower_stack.size() > 1) {
          auto child = lower_stack.top();
          lower_stack.pop();
          auto parent = lower_stack.top();
          parent->AddChild(child);
        }
      } break;
      default:
        break;
    }
    cur_pos = next_pos + 1;
  }

  BaseLowerNodePtr lower_root;
  while (!lower_stack.empty()) {
    if (lower_stack.size() > 1) {
      auto child = lower_stack.top();
      lower_stack.pop();
      auto parent = lower_stack.top();
      parent->AddChild(child);
    } else {
      lower_root = lower_stack.top();
      lower_stack.pop();
    }
  }
  return lower_root;
}

Module LowerCompositeToModule(const std::string &target, bool poly, const std::string &segment_tree_str,
                              const Map<std::string, NodeRef> &segment_infos) {
  auto build_str = std::string(kModule) + "0[" + segment_tree_str + "]";
  auto build_root = std::dynamic_pointer_cast<ModuleLowerNode>(
    ConstructLowerTree(GetRealTarget(target), poly, build_str, segment_infos));
  build_root->Process();
  return build_root->GetModule();
}

NodeRef TuneComposite(const std::string &target, bool poly, const std::string &segment_tree_str,
                      const Map<std::string, NodeRef> &segment_infos) {
  auto build_str = std::string(kTune) + "0[" + segment_tree_str + "]";
  auto lower_root = ConstructLowerTree(GetRealTarget(target), poly, build_str, segment_infos);
  lower_root->Run();
  return lower_root->Node();
}
}  // namespace lower

TVM_REGISTER_GLOBAL("lower_composite_to_module").set_body_typed(lower::LowerCompositeToModule);
TVM_REGISTER_GLOBAL("tune_composite").set_body_typed(lower::TuneComposite);
}  // namespace akg
