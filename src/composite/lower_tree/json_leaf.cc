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

#include "composite/lower_tree/json_leaf.h"

#include <string>
#include <utility>
#include "composite/extract_build_info.h"

namespace akg {
namespace lower {
namespace {
constexpr auto kAttrs = "attrs";
constexpr auto kJsonStr = "json_str";
constexpr auto kKernelName = "kernel_name";
}  // namespace
void JsonLowerLeaf::Lower(StageType s) {
  auto info = GenBuildInfo(attrs_);
  attrs_.Set(kOriginKernelName, Expr(origin_kernel_name_));
  Map<Tensor, Map<std::string, NodeRef>> tensor_attrs_map;
  for (auto it : info_.opt.tensor_attrs) {
    tensor_attrs_map.Set(it.first, it.second);
  }
  attrs_.Set(kTensorAttrs, tensor_attrs_map);
  std::string process = GetProcess(String2Json(json_str_));
  if (attrs_.count("target_option")) {
    CHECK(attrs_["target_option"]->IsInstance<StringImm>());
    process = process + " " + attrs_["target_option"].as<StringImm>()->value;
  }
  data_ = LowerDataNode::make(GetScheduleWithBuildInfo(info), info.args, info.in_binds, attrs_, process,
                              info.kernel_name, GetConfig(), polyhedral_);
  current_stage_ = StageType::Begin;
}

BuildInfo JsonLowerLeaf::GenBuildInfo(Map<std::string, NodeRef> &attrs) {
  ExtractBuildInfo(String2Json(json_str_), info_);
  if (attrs.find(kKernelName) != attrs.end()) {
    CHECK(attrs[kKernelName]->IsInstance<StringImm>());
    info_.kernel_name = attrs[kKernelName].as<StringImm>()->value;
  }
  origin_kernel_name_ = info_.kernel_name;  // For kernel build
  return info_;
}

BaseLowerNodePtr CreateJsonLowerLeaf(const std::string &target, bool poly,
                                     const Map<std::string, NodeRef> &construct_infos) {
  auto json_str = construct_infos[kJsonStr].as<StringImm>()->value;
  auto attrs = Downcast<Map<std::string, NodeRef>>(construct_infos[kAttrs]);
  return std::make_shared<JsonLowerLeaf>(target, json_str, attrs, poly);
}

REG_NODE_CREATOR(kLlvm, kP, CreateJsonLowerLeaf);
REG_NODE_CREATOR(kCuda, kP, CreateJsonLowerLeaf);
REG_NODE_CREATOR(kCce, kP, CreateJsonLowerLeaf);
}  // namespace lower
}  // namespace akg
