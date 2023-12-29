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
#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_BASE_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_BASE_NODE_H_
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <pass/utils.h>
#include "build_module.h"
#include "codegen/lower.h"
#include "codegen/pass_mgr.h"
#include "codegen/stage_lower.h"
#include "composite/utils/dump.h"
#include "composite/utils/util.h"

namespace akg {
namespace lower {
#define DUMP_ORIGIN_IR(dump_manager, arg0) dump_manager.DumpStmt("Origin", arg0)
#define TRANSFORM_AND_TRY_DUMP(dump_manager, out0, call, arg0, ...) \
  do {                                                              \
    out0 = call(arg0, ##__VA_ARGS__);                               \
    dump_manager.DumpStmt(#call, out0);                             \
  } while (0)

constexpr auto kCce = "cce";
constexpr auto kCuda = "cuda";
constexpr auto kLlvm = "llvm";

constexpr auto kKernelInputs = "kernel_inputs";
constexpr auto kKernelOutputs = "kernel_outputs";

constexpr auto kKernelNamePosfix = "kernel_name_postfix";
constexpr auto kOriginKernelName = "origin_kernel_name";

constexpr auto kExtraAttrs = "extra_attrs";

constexpr auto kFoldDim = "fold_dim";

Schedule GetScheduleWithBuildInfo(const BuildInfo &info);

class BaseLowerNode;
class JsonLowerLeaf;
using BaseLowerNodePtr = std::shared_ptr<BaseLowerNode>;

class LowerRunner {
 public:
  LowerRunner() = default;
  virtual ~LowerRunner() = default;
  virtual void Lower(StageType s) = 0;
};

class BaseLowerNode : public LowerRunner {
 public:
  BaseLowerNode() = default;
  explicit BaseLowerNode(const std::string &target) : target_(target) { name_ = __FUNCTION__; }
  virtual ~BaseLowerNode() = default;

  void Run(StageType s = StageType::Unknown) {
    runner_->Lower(s);
  }

  void Run(const BaseLowerNode *parent) {
    if (current_stage_ == StageType::Unknown ||
        (parent->Data() && StageTypeLT(target_, current_stage_, parent->entrance_stage_))) {
      Run(parent->entrance_stage_);
    }
  }

  void Lower(StageType s) override {}

  void Decorate(const std::function<void(BaseLowerNode*, LowerRunner*, StageType)> &fn);
  void VisitLeaf(const std::function<void(JsonLowerLeaf *)> &fn);

  NodeRef Node() const { return node_ref_; }
  LowerData Data() const { return data_; }
  void AddChild(BaseLowerNodePtr child) { children_.push_back(child); }

  StageType entrance_stage_{StageType::Begin};
  StageType current_stage_{StageType::Unknown};

 protected:
  std::string target_;
  std::string name_;
  std::vector<BaseLowerNodePtr> children_;
  NodeRef node_ref_;
  LowerData data_;
  LowerRunner *runner_{this};
  std::vector<std::unique_ptr<LowerRunner>> decorators_;
};

using LowerNodeCreateFunc =
  std::function<BaseLowerNodePtr(const std::string &, bool, const Map<std::string, NodeRef> &)>;
class LowerNodeCreatorManager {
 public:
  static LowerNodeCreatorManager &Instance();
  void RegisterNodeCreator(const std::string &target, const std::string &node_type, LowerNodeCreateFunc func);
  LowerNodeCreateFunc GetLowerNodeCreator(const std::string &target, const std::string &node_type);

 private:
  LowerNodeCreatorManager() = default;
  ~LowerNodeCreatorManager() = default;
  LowerNodeCreatorManager(const LowerNodeCreatorManager &) = delete;
  LowerNodeCreatorManager &operator=(const LowerNodeCreatorManager &) = delete;
  std::unordered_map<std::string, std::unordered_map<std::string, LowerNodeCreateFunc>> creator_funcs_;
};

struct NodeCreatorRegister {
  NodeCreatorRegister(const std::string &target, const std::string &node_type, LowerNodeCreateFunc func) {
    LowerNodeCreatorManager::Instance().RegisterNodeCreator(target, node_type, func);
  }
};
#define REG_NODE_CREATOR(target, node_type, func) REG_LOWER_BASE(NodeCreatorRegister, target, node_type, func)
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_BASE_NODE_H_
