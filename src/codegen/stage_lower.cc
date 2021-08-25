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
#include "codegen/stage_lower.h"
#include <algorithm>
#include <functional>

namespace akg {
namespace lower {
StageManager &StageManager::Instance() {
  static StageManager instance;
  return instance;
}

void StageManager::RegisterStage(const std::string &target, StageType stage_type, const std::string &name,
                                 std::function<StageResult(Stmt &, LowerData &)> func) {
  if (stages_.find(target) == stages_.end()) {
    stages_.insert({target, std::vector<Stage>()});
  }
  if (std::any_of(stages_[target].cbegin(), stages_[target].cend(),
                  [&stage_type](const Stage &s) { return s.type == stage_type; })) {
    LOG(ERROR) << "Stage " << static_cast<int16_t>(stage_type) << " for " << target << " is all ready exist!";
    return;
  }

  stages_[target].emplace_back(stage_type, name, func);
}

void StageManager::RegisterFilter(
  const std::string &target,
  std::function<std::vector<Stage>(const LowerData &, StageType, StageType, const std::vector<Stage> &)> func) {
  if (stage_filters_.find(target) != stage_filters_.end()) {
    LOG(ERROR) << "Filter for " << target << " is all ready exist!";
    return;
  }

  stage_filters_.insert({target, func});
}

std::vector<Stage> StageManager::GetStages(const LowerData &data, StageType begin, StageType end) {
  CHECK(stages_.find(data->target) != stages_.end()) << GetErrorHint(data->target);

  auto bg_offset = GetIndexOfStageType(data->target, begin);
  auto end_offset = GetIndexOfStageType(data->target, end);
  CHECK(bg_offset <= end_offset);

  std::vector<Stage> stages(stages_[data->target].begin() + bg_offset, stages_[data->target].begin() + end_offset + 1);
  if (stage_filters_.find(data->target) != stage_filters_.end()) {
    return stage_filters_[data->target](data, begin, end, stages);
  }

  return stages;
}

size_t StageManager::GetIndexOfStageType(const std::string &target, StageType type) {
  CHECK(stages_.find(target) != stages_.end()) << GetErrorHint(target);

  for (size_t i = 0; i < stages_[target].size(); ++i) {
    if (stages_[target][i].type == type) {
      return i;
    }
  }
  CHECK(0) << "Unsupport stage " << static_cast<int16_t>(type) << " for " << target;
  return 0;
}

Stage StageManager::GetStageByType(const std::string &target, StageType type) {
  CHECK(stages_.find(target) != stages_.end()) << GetErrorHint(target);
  for (auto stage : stages_[target]) {
    if (stage.type == type) {
      return stage;
    }
  }
  CHECK(0) << "Unsupport stage " << static_cast<int16_t>(type) << " for " << target;
  return Stage();
}

StageType StageManager::NextStageType(const std::string &target, StageType type) {
  CHECK(stages_.find(target) != stages_.end()) << GetErrorHint(target);
  auto iter =
    std::find_if(stages_[target].cbegin(), stages_[target].cend(), [&type](const Stage &s) { return s.type == type; });
  CHECK(iter != stages_[target].cend());
  StageType res = iter->type;
  ++iter;
  if (iter != stages_[target].cend()) {
    res = iter->type;
  }

  return res;
}

bool StageLower::SkipTo(StageType to) {
  size_t cur_index = StageManager::Instance().GetIndexOfStageType(data_->target, cur_stage_);
  size_t to_index = StageManager::Instance().GetIndexOfStageType(data_->target, to);
  if (to_index <= cur_index) {
    LOG(WARNING) << "The stage (" << StageManager::Instance().GetStageByType(data_->target, to)
                 << ") want to skip to is behind the current stage("
                 << StageManager::Instance().GetStageByType(data_->target, cur_stage_) << ")!";
    return false;
  }
  cur_stage_ = to;
  return true;
}

StageLower &StageLower::RunTo(StageType to) {
  if (done_) {
    LOG(WARNING) << "The compil>e is done, cannot run "
                 << StageManager::Instance().GetStageByType(data_->target, cur_stage_) << " to "
                 << StageManager::Instance().GetStageByType(data_->target, to);
    return *this;
  }

  if (data_->attrs.defined()) {
    g_attrs = data_->attrs;
  }

  Stmt stmt = node_ref_.defined() ? Downcast<Stmt>(node_ref_) : Stmt();
  auto stages = StageManager::Instance().GetStages(data_, cur_stage_, to);
  for (auto stage : stages) {
    if (done_) {
      break;
    }
    LOG(INFO) << "Run stage " << stage;
    auto res = stage.func(stmt, data_);
    node_ref_ = res.first;
    if (res.second) {
      done_ = true;
    } else {
      stmt = Downcast<Stmt>(res.first);
      CHECK(stmt);
    }
  }

  // Write back global attrs to data's attrs, to make sure next `RunTo` will get right attribute.
  for (auto iter : g_attrs) {
    data_->attrs.Set(iter.first, iter.second);
  }

  cur_stage_ = StageManager::Instance().NextStageType(data_->target, to);
  return *this;
}
}  // namespace lower
}  // namespace akg
