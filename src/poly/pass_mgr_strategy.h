/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef POLY_PASS_MGR_STRATEGY_H_
#define POLY_PASS_MGR_STRATEGY_H_

#include "poly/schedule_pass.h"
#include "poly/dump_log.h"

#include "poly/schedule_pass/init_schedule.h"
#include "poly/schedule_pass/compute_schedule.h"

namespace akg {
namespace ir {
namespace poly {

class PassMgrStrategy {
 public:
  explicit PassMgrStrategy(ScopInfo &scop_info) : scop_info_(scop_info) {}

  void RegisterPass(std::shared_ptr<SchedulePass> pass) {
    CHECK(pass);
    passes_.emplace_back(std::move(pass));
  }
  void RegisterNormalizationPasses() { RegisterPass(std::make_shared<InitSchedule>(pass_info_, scop_info_)); }
  void RegisterSchedulingPasses() { RegisterPass(std::make_shared<ComputeSchedule>(pass_info_, scop_info_)); }
  virtual void RegisterTilingPasses() = 0;   // each backend has different achievement
  virtual void RegisterMemPromPasses() = 0;  // each backend has different achievement
  virtual void RegisterPasses() = 0;
  const std::vector<std::shared_ptr<SchedulePass>> &GetPasses() const { return passes_; };

  virtual ~PassMgrStrategy() = default;

  ScopInfo &scop_info_;
  PassInfo pass_info_;

 protected:
  std::vector<std::shared_ptr<SchedulePass>> passes_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_PASS_MGR_STRATEGY_H_