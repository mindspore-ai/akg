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

/*
 * This tool is use for stage lower.
 *
 * Example case: Run to `Poly`, then modify the node_ref, and run to `End` again:
 *
 * ```cpp
 * LowerData data = LowerDataNode::make();
 * data->xxx = xxx; // Setup data option, for example, setup target: data->target = "cuda";
 *
 * auto stage_lower = StageLower(data);
 * stage_lower.RunTo(StageType::Poly);
 * stage_lower.ApplyMutator([](NodeRef &node_ref, LowerData &data) {
 *   // Mutate the node_ref, data or both.
 * });
 * stage_lower.RunTo(StageType::End);
 * NodeRef node_ref = stage_lower.Node();
 * ```
 */

#ifndef CODEGEN_STAGE_LOEWR_H_
#define CODEGEN_STAGE_LOEWR_H_
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "build_module.h"
#include "codegen/lower.h"

namespace akg {
extern AttrMap g_attrs;
extern CsrMap g_csr;
namespace lower {
enum class StageType : int16_t {
  Begin = 0,
  Tuning,
  Poly,
  BeforeFlattern,
  Flattern,
  MultiCore,
  BeforeRewrite,
  Rewrite,
  BeforeLowerFunc,
  End,
  Unknown
};

class Stage {
 public:
  Stage() = default;
  Stage(StageType type, const std::string &name, std::function<StageResult(Stmt &, LowerData &)> func)
      : type(type), name(name), func(func) {}
  ~Stage() = default;

  StageType type{StageType::Unknown};
  std::string name{"Unknown"};
  std::function<StageResult(Stmt &, LowerData &)> func;
};
inline std::ostream &operator<<(std::ostream &os, Stage s) { return os << s.name; }

class StageManager {
 public:
  static StageManager &Instance();
  void RegisterStage(const std::string &target, StageType stage_type, const std::string &name,
                     std::function<StageResult(Stmt &, LowerData &)> func);
  void RegisterFilter(
    const std::string &target,
    std::function<std::vector<Stage>(const LowerData &, StageType, StageType, const std::vector<Stage> &)> func);
  std::vector<Stage> GetStages(const LowerData &data, StageType begin, StageType end);
  size_t GetIndexOfStageType(const std::string &target, StageType stage);
  Stage GetStageByType(const std::string &target, StageType type);
  StageType PreStageType(const std::string &target, StageType type);
  StageType NextStageType(const std::string &target, StageType type);

 private:
  StageManager() = default;
  ~StageManager() = default;
  StageManager(const StageManager &) = delete;
  StageManager &operator=(const StageManager &) = delete;
  std::unordered_map<std::string, std::vector<Stage>> stages_;
  std::unordered_map<
    std::string, std::function<std::vector<Stage>(const LowerData &, StageType, StageType, const std::vector<Stage> &)>>
    stage_filters_;
};

class StageLower {
 public:
  StageLower() = delete;
  explicit StageLower(const LowerData &data) : data_(data) {}
  StageLower(const LowerData &data, const NodeRef &node_ref, StageType start_stage = StageType::Begin)
      : cur_stage_(start_stage), data_(data), node_ref_(node_ref) {}
  ~StageLower() {}

  bool SkipTo(StageType to);

  // Run stage [cur_stage_, to].
  StageLower &RunTo(StageType to = StageType::End);
  StageLower &ApplyMutator(std::function<NodeRef(NodeRef &, LowerData &)> func) {
    node_ref_ = func(node_ref_, data_);
    return *this;
  };
  void SetNode(const NodeRef &node_ref) { node_ref_ = node_ref; }
  NodeRef Node() { return node_ref_; }
  LowerData Data() { return data_; }
  StageType GetCurStage() { return cur_stage_; }

 private:
  StageType cur_stage_{StageType::Begin};  // Next run will begin with.
  LowerData data_;
  NodeRef node_ref_;
  bool done_{false};
};

inline bool StageTypeLT(const std::string &target, StageType a, StageType b) {
  return StageManager::Instance().GetIndexOfStageType(target, a) <
         StageManager::Instance().GetIndexOfStageType(target, b);
}
inline bool StageTypeLE(const std::string &target, StageType a, StageType b) {
  return StageManager::Instance().GetIndexOfStageType(target, a) <=
         StageManager::Instance().GetIndexOfStageType(target, b);
}
inline bool StageTypeGT(const std::string &target, StageType a, StageType b) {
  return StageManager::Instance().GetIndexOfStageType(target, a) >
         StageManager::Instance().GetIndexOfStageType(target, b);
}
inline bool StageTypeGE(const std::string &target, StageType a, StageType b) {
  return StageManager::Instance().GetIndexOfStageType(target, a) >=
         StageManager::Instance().GetIndexOfStageType(target, b);
}

struct StageRegister {
  StageRegister(const std::string &target, StageType stage_type, const std::string &name,
                std::function<StageResult(Stmt &, LowerData &)> func) {
    StageManager::Instance().RegisterStage(target, stage_type, name, func);
  }
};
#define REG_STAGE_LOWER(target, type, name, func) REG_LOWER_BASE(StageRegister, target, type, name, func)

struct FilterRegister {
  FilterRegister(
    const std::string &target,
    std::function<std::vector<Stage>(const LowerData &, StageType, StageType, const std::vector<Stage> &)> func) {
    StageManager::Instance().RegisterFilter(target, func);
  }
};
#define REG_FILTER_LOWER(target, func) REG_LOWER_BASE(FilterRegister, target, func)

}  // namespace lower
}  // namespace akg
#endif  // CODEGEN_STAGE_LOEWR_H_
