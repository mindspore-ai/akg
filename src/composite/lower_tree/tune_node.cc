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

#include "composite/lower_tree/tune_node.h"

#include <sstream>
#include <vector>
#include <pass/utils.h>
#include "build_module.h"
#include "codegen/lower.h"
#include "codegen/pass_mgr.h"
#include "codegen/stage_lower.h"
#include "composite/utils/dump.h"
#include "composite/parser.h"
#include "composite/utils/util.h"
#include "composite/lower_tree/base_node.h"
#include "composite/lower_tree/json_leaf.h"

namespace akg {
namespace lower {
namespace {
constexpr auto kArgs = "args";
constexpr auto kEnableTune = "enable_tune";
constexpr auto kRetMode = "ret_mode";
constexpr auto kTuning = "tuning";
void ModifyTuneInfo(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos, BuildInfo *info) {
  if (forward_infos.find(kEnableTune) == forward_infos.end()) {
    return;
  }
  if (attrs.find(kFoldDim) != attrs.end()) {
    info->opt.fold_dim = GetBoolValueFromMap(attrs, kFoldDim);
  }
}

void ModifyTuneBackwardArgs(Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> &attrs, BuildInfo &info,
                            Map<std::string, NodeRef> *backward_infos) {
  if (forward_infos.find(kEnableTune) == forward_infos.end()) {
    return;
  }

  if (attrs.find(kRetMode) != attrs.end()) {
    backward_infos->Set(kRetMode, Expr(1));
  }
  backward_infos->Set(kArgs, info.args);
}

void ModifyTuneData(Map<std::string, NodeRef> &forward_infos, LowerData &data) {
  if (forward_infos.find(kEnableTune) == forward_infos.end()) {
    return;
  }

  if (data->attrs.find(kRetMode) != data->attrs.end()) {
    if (data->attrs[kRetMode]->IsInstance<IntImm>() && data->attrs[kRetMode].as<IntImm>()->value == 1) {
      data->tuning = false;
    }
  } else {
    data->tuning = data->attrs.find(kTuning) != data->attrs.end();
  }
}
}  // namespace

void TuneLowerNode::ExcuteImpl(StageType stage) {
  CHECK(children_.size() == 1);  // Only support 1 child now.
  auto &child = children_[0];

  Map<std::string, NodeRef> forward_infos;
  forward_infos.Set(kEnableTune, Expr(1));

  Excute(child, forward_infos);

  auto backward_infos = child->BackwardInfos();
  if (backward_infos.find(kRetMode) != backward_infos.end()) {
    CHECK(backward_infos.find(kArgs) != backward_infos.end());
    node_ref_ = Array<NodeRef>({child->Node(), backward_infos[kArgs]});
  } else {
    node_ref_ = child->Node();
  }
}

#define REG_TUNE_FUNC(target)                                         \
  REG_INFO_FUNC_BEFORE(target, "TuneLowerNode", ModifyTuneInfo);      \
  REG_BACKWARD_FUNC(target, "TuneLowerNode", ModifyTuneBackwardArgs); \
  REG_DATA_FUNC(target, "TuneLowerNode", ModifyTuneData)

REG_TUNE_FUNC(kCuda);
REG_TUNE_FUNC(kCce);
REG_TUNE_FUNC(kLlvm);

BaseLowerNodePtr CreateTuneLowerNode(const std::string &target, bool,
                                     const Map<std::string, NodeRef> &construct_infos) {
  bool get_stmt = false;
  if (construct_infos.find(kGetStmt) != construct_infos.end()) {
    get_stmt = GetBoolValueFromMap(construct_infos, kGetStmt);
  }
  return std::make_shared<TuneLowerNode>(target, get_stmt);
}

REG_NODE_CREATOR(kLlvm, kTune, CreateTuneLowerNode);
REG_NODE_CREATOR(kCuda, kTune, CreateTuneLowerNode);
REG_NODE_CREATOR(kCce, kTune, CreateTuneLowerNode);
}  // namespace lower
}  // namespace akg
