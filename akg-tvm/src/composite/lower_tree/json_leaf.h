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
  void Lower(StageType s) override;

  Map<std::string, NodeRef>& Attrs() { return attrs_; }
  std::string& JsonDesc() { return json_str_; }
  LowerData& MutateData()  { return data_; }

  BuildInfo info_;

 private:
  BuildInfo GenBuildInfo(Map<std::string, NodeRef> &attrs);

  std::string json_str_;
  Map<std::string, NodeRef> attrs_;
  bool polyhedral_{true};
  std::string origin_kernel_name_;
};
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_JSON_LEAF_H_
