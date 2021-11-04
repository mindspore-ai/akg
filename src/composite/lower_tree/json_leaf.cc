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
void JsonLowerLeaf::ExcuteImpl(StageType s) {
  if (forward_infos_.find(kCatch) != forward_infos_.end()) {
    backward_infos_.Set(kBlockJsons, Expr(json_str_));
    backward_infos_.Set(kBlockAttrs, attrs_);
    return;
  }

  auto info = GenBuildInfo(attrs_);
  ModifyAttrs(forward_infos_, info, &attrs_);
  ModifyBackwardInfos(forward_infos_, attrs_, info, &backward_infos_);
  data_ = LowerDataNode::make(GetScheduleWithBuildInfo(info), info.args, info.in_binds, attrs_,
                              GetProcess(String2Json(json_str_)), info.kernel_name, GetConfig(), polyhedral_);
  ModifyData(forward_infos_, data_);
  current_stage_ = StageType::Begin;
}

void JsonLowerLeaf::ModifyInfoBeforeExtract(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos,
                                            BuildInfo *info) {
  auto func_pairs = JsonProc::Instance().GetProcInfoFuncs(target_, JsonStage::BeforeExtra);
  for (auto func_pair : func_pairs) {
    auto &func = func_pair.second;
    func(attrs, forward_infos, info);
  }
}

void JsonLowerLeaf::ModifyInfoAfterExtract(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos,
                                           BuildInfo *info) {
  auto func_pairs = JsonProc::Instance().GetProcInfoFuncs(target_, JsonStage::AfterExtra);
  for (auto func_pair : func_pairs) {
    auto &func = func_pair.second;
    func(attrs, forward_infos, info);
  }
}

BuildInfo JsonLowerLeaf::GenBuildInfo(Map<std::string, NodeRef> &attrs) {
  GetAllAttrs(attrs);
  BuildInfo info;

  ModifyInfoBeforeExtract(attrs, forward_infos_, &info);

  ExtractBuildInfo(String2Json(json_str_), info);

  if (attrs.find(kKernelName) != attrs.end()) {
    CHECK(attrs[kKernelName]->IsInstance<StringImm>());
    info.kernel_name = attrs[kKernelName].as<StringImm>()->value;
  }
  origin_kernel_name_ = info.kernel_name;  // For kernel build
  if (forward_infos_.find(kKernelNamePosfix) != forward_infos_.end()) {
    CHECK(forward_infos_[kKernelNamePosfix]->IsInstance<StringImm>());
    info.kernel_name += "_" + forward_infos_[kKernelNamePosfix].as<StringImm>()->value;  // For pass dump
  }

  ModifyInfoAfterExtract(attrs, forward_infos_, &info);

  return info;
}

void JsonLowerLeaf::GetAllAttrs(Map<std::string, NodeRef> &attrs) {
  if (forward_infos_.find(kExtraAttrs) != forward_infos_.end()) {
    auto extra_attrs = Downcast<Map<std::string, NodeRef>>(forward_infos_[kExtraAttrs]);
    for (auto attr : extra_attrs) {
      attrs.Set(attr.first, attr.second);
    }
  }
}

void JsonLowerLeaf::ModifyAttrs(Map<std::string, NodeRef> &forward_infos, BuildInfo &info,
                                Map<std::string, NodeRef> *attrs) {
  attrs->Set(kOriginKernelName, Expr(origin_kernel_name_));

  auto func_pairs = JsonProc::Instance().GetProcAttrsFuncs(target_, JsonStage::ModifyAttrs);
  for (auto func_pair : func_pairs) {
    auto &func = func_pair.second;
    func(forward_infos, info, attrs);
  }
}

void JsonLowerLeaf::ModifyBackwardInfos(Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> &attrs,
                                        BuildInfo &info, Map<std::string, NodeRef> *backward_infos) {
  auto func_pairs = JsonProc::Instance().GetProcBackwardFuncs(target_, JsonStage::ModifyBackwardInfos);
  for (auto func_pair : func_pairs) {
    auto &func = func_pair.second;
    func(forward_infos, attrs, info, backward_infos);
  }
}

void JsonLowerLeaf::ModifyData(Map<std::string, NodeRef> &forward_infos, LowerData &data) {
  auto func_pairs = JsonProc::Instance().GetProcDataFuncs(target_, JsonStage::ModifyData);
  for (auto func_pair : func_pairs) {
    auto &func = func_pair.second;
    func(forward_infos, data);
  }
}

JsonProc &JsonProc::Instance() {
  static JsonProc instance;
  return instance;
}

void JsonProc::RegisterInfoFunc(const std::string &target, JsonStage js, const std::string &prov, ProcInfoFunc func) {
  auto key = ProcKey(target, js);
  if (info_funcs_.find(key) == info_funcs_.end()) {
    info_funcs_.emplace(key, ProcInfoFuncList());
  }
  info_funcs_[key].emplace_back(std::make_pair(prov, func));
}

ProcInfoFuncList JsonProc::GetProcInfoFuncs(const std::string &target, JsonStage js) {
  auto key = ProcKey(target, js);
  if (info_funcs_.find(key) == info_funcs_.end()) {
    return ProcInfoFuncList();
  }
  return info_funcs_[key];
}

void JsonProc::RegisterAttrFunc(const std::string &target, JsonStage js, const std::string &prov, ProcAttrFunc func) {
  auto key = ProcKey(target, js);
  if (attrs_funcs_.find(key) == attrs_funcs_.end()) {
    attrs_funcs_.emplace(key, ProcAttrFuncList());
  }
  attrs_funcs_[key].emplace_back(std::make_pair(prov, func));
}
ProcAttrFuncList JsonProc::GetProcAttrsFuncs(const std::string &target, JsonStage js) {
  auto key = ProcKey(target, js);
  if (attrs_funcs_.find(key) == attrs_funcs_.end()) {
    return ProcAttrFuncList();
  }
  return attrs_funcs_[key];
}

void JsonProc::RegisterBackwardFunc(const std::string &target, JsonStage js, const std::string &prov,
                                    ProcBackwardFunc func) {
  auto key = ProcKey(target, js);
  if (backward_funcs_.find(key) == backward_funcs_.end()) {
    backward_funcs_.emplace(key, ProcBackwardFuncList());
  }
  backward_funcs_[key].emplace_back(std::make_pair(prov, func));
}
ProcBackwardFuncList JsonProc::GetProcBackwardFuncs(const std::string &target, JsonStage js) {
  auto key = ProcKey(target, js);
  if (backward_funcs_.find(key) == backward_funcs_.end()) {
    return ProcBackwardFuncList();
  }
  return backward_funcs_[key];
}

void JsonProc::RegisterDataFunc(const std::string &target, JsonStage js, const std::string &prov, ProcDataFunc func) {
  auto key = ProcKey(target, js);
  if (data_funcs_.find(key) == data_funcs_.end()) {
    data_funcs_.emplace(key, ProcDataFuncList());
  }
  data_funcs_[key].emplace_back(std::make_pair(prov, func));
}
ProcDataFuncList JsonProc::GetProcDataFuncs(const std::string &target, JsonStage js) {
  auto key = ProcKey(target, js);
  if (data_funcs_.find(key) == data_funcs_.end()) {
    return ProcDataFuncList();
  }
  return data_funcs_[key];
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
