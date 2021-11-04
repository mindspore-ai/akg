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

#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_TOT_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_TOT_NODE_H_
#include <sstream>
#include <vector>
#include <pass/utils.h>
#include "build_module.h"
#include "codegen/lower.h"
#include "codegen/pass_mgr.h"
#include "codegen/stage_lower.h"
#include "composite/utils/dump.h"
#include "composite/utils/util.h"
#include "pass/recover_tot.h"
#include "pass/replace_tot.h"
#include "composite/lower_tree/base_node.h"

namespace akg {
namespace lower {
constexpr auto kTot = "Tot";
constexpr auto kNoInlineIndeed = "noinline_indeed";
class TotLowerNode : public BaseLowerNode {
 public:
  explicit TotLowerNode(const std::string &target) : BaseLowerNode(target) {
    name_ = __FUNCTION__;
    entrance_stage_ = StageManager::Instance().PreStageType(target, StageType::Poly);
  }
  ~TotLowerNode() override {}

  void ExcuteImpl(StageType stage) override;
};
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_TOT_NODE_H_
