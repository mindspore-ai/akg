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
#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_JSON_LEAF_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_JSON_LEAF_H_
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <pass/utils.h>
#include "picojson.h"
#include "build_module.h"
#include "codegen/lower.h"
#include "codegen/pass_mgr.h"
#include "codegen/stage_lower.h"
#include "composite/utils/dump.h"
#include "composite/utils/util.h"
#include "composite/lower_tree/base_node.h"

namespace akg {
namespace lower {
constexpr auto kP = "P";

class JsonLowerLeaf : public BaseLowerNode {
 public:
  JsonLowerLeaf(const std::string &target, const std::string &json_str, const Map<std::string, NodeRef> &attrs,
                bool poly = true)
      : BaseLowerNode(target), json_str_(json_str), attrs_(attrs), polyhedral_(poly) {
    name_ = __FUNCTION__;
  }
  ~JsonLowerLeaf() = default;
  void ExcuteImpl(StageType s) override;

 private:
  BuildInfo GenBuildInfo(Map<std::string, NodeRef> &attrs);
  void GetAllAttrs(Map<std::string, NodeRef> &attrs);
  void ModifyAttrs(Map<std::string, NodeRef> &forward_infos, BuildInfo &info, Map<std::string, NodeRef> *attrs);
  void ModifyBackwardInfos(Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> &attrs, BuildInfo &info,
                           Map<std::string, NodeRef> *backward_infos);
  void ModifyData(Map<std::string, NodeRef> &forward_infos, LowerData &data);
  void ModifyInfoBeforeExtract(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos,
                               BuildInfo *info);
  void ModifyInfoAfterExtract(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos,
                              BuildInfo *info);

  std::string json_str_;
  Map<std::string, NodeRef> attrs_;
  bool polyhedral_{true};
  std::string origin_kernel_name_;
};

enum class JsonStage : int16_t { BeforeExtra, AfterExtra, ModifyAttrs, ModifyBackwardInfos, ModifyData };
using ProcInfoFunc = std::function<void(Map<std::string, NodeRef> &, Map<std::string, NodeRef> &, BuildInfo *)>;
using ProcAttrFunc = std::function<void(Map<std::string, NodeRef> &, BuildInfo &, Map<std::string, NodeRef> *)>;
using ProcBackwardFunc = std::function<void(Map<std::string, NodeRef> &, Map<std::string, NodeRef> &, BuildInfo &,
                                            Map<std::string, NodeRef> *)>;
using ProcDataFunc = std::function<void(Map<std::string, NodeRef> &, LowerData &)>;

using ProcInfoFuncList = std::vector<std::pair<std::string, ProcInfoFunc>>;
using ProcAttrFuncList = std::vector<std::pair<std::string, ProcAttrFunc>>;
using ProcBackwardFuncList = std::vector<std::pair<std::string, ProcBackwardFunc>>;
using ProcDataFuncList = std::vector<std::pair<std::string, ProcDataFunc>>;

class JsonProc {
 public:
  static JsonProc &Instance();
  void RegisterInfoFunc(const std::string &target, JsonStage js, const std::string &prov, ProcInfoFunc func);
  void RegisterAttrFunc(const std::string &target, JsonStage js, const std::string &prov, ProcAttrFunc func);
  void RegisterBackwardFunc(const std::string &target, JsonStage js, const std::string &prov, ProcBackwardFunc func);
  void RegisterDataFunc(const std::string &target, JsonStage js, const std::string &prov, ProcDataFunc func);
  ProcInfoFuncList GetProcInfoFuncs(const std::string &target, JsonStage js);
  ProcAttrFuncList GetProcAttrsFuncs(const std::string &target, JsonStage js);
  ProcBackwardFuncList GetProcBackwardFuncs(const std::string &target, JsonStage js);
  ProcDataFuncList GetProcDataFuncs(const std::string &target, JsonStage js);

 private:
  JsonProc() = default;
  ~JsonProc() = default;
  JsonProc(const JsonProc &) = delete;
  JsonProc &operator=(const JsonProc &) = delete;

  struct ProcKey {
    ProcKey(const std::string &target, JsonStage js) : target(target), json_stage(js) {}
    ~ProcKey() = default;
    std::string target;
    JsonStage json_stage;
    bool operator<(const ProcKey &p) const {
      if (target < p.target) {
        return true;
      } else {
        return json_stage < p.json_stage;
      }
    }
  };

  std::map<ProcKey, ProcInfoFuncList> info_funcs_;
  std::map<ProcKey, ProcAttrFuncList> attrs_funcs_;
  std::map<ProcKey, ProcBackwardFuncList> backward_funcs_;
  std::map<ProcKey, ProcDataFuncList> data_funcs_;
};

struct JsonProcessRegister {
  JsonProcessRegister(const std::string &target, JsonStage js, const std::string &prov, ProcInfoFunc func) {
    JsonProc::Instance().RegisterInfoFunc(target, js, prov, func);
  }
  JsonProcessRegister(const std::string &target, JsonStage js, const std::string &prov, ProcAttrFunc func) {
    JsonProc::Instance().RegisterAttrFunc(target, js, prov, func);
  }
  JsonProcessRegister(const std::string &target, JsonStage js, const std::string &prov, ProcBackwardFunc func) {
    JsonProc::Instance().RegisterBackwardFunc(target, js, prov, func);
  }
  JsonProcessRegister(const std::string &target, JsonStage js, const std::string &prov, ProcDataFunc func) {
    JsonProc::Instance().RegisterDataFunc(target, js, prov, func);
  }
};
#define REG_INFO_FUNC_BEFORE(target, prov, func) \
  REG_LOWER_BASE(JsonProcessRegister, target, JsonStage::BeforeExtra, prov, func)
#define REG_INFO_FUNC_AFTER(target, prov, func) \
  REG_LOWER_BASE(JsonProcessRegister, target, JsonStage::AfterExtra, prov, func)
#define REG_ATTR_FUNC(target, prov, func) \
  REG_LOWER_BASE(JsonProcessRegister, target, JsonStage::ModifyAttrs, prov, func)
#define REG_BACKWARD_FUNC(target, prov, func) \
  REG_LOWER_BASE(JsonProcessRegister, target, JsonStage::ModifyBackwardInfos, prov, func)
#define REG_DATA_FUNC(target, prov, func) REG_LOWER_BASE(JsonProcessRegister, target, JsonStage::ModifyData, prov, func)
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_JSON_LEAF_H_
