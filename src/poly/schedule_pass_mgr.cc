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

#include "poly/schedule_pass_mgr.h"

namespace akg {
namespace ir {
namespace poly {

const std::vector<std::shared_ptr<SchedulePass>> &SchedulePassMgr::GetSchedulePasses() const {
  return schedule_passes_;
}

void SchedulePassMgr::RegisterPass(std::shared_ptr<SchedulePass> pass) {
  CHECK(pass);
  schedule_passes_.push_back(pass);
}

isl::schedule SchedulePassMgr::Run(const isl::schedule &sch) {
  CHECK(sch);
  return Run(sch, schedule_passes_);
}

isl::schedule SchedulePassMgr::Run(const isl::schedule &sch, const std::vector<std::shared_ptr<SchedulePass>> &passes) {
  CHECK(sch);

  std::chrono::high_resolution_clock::time_point timer_start;
  scop_info_.ClearTimeRecords();

  auto final_sch = sch;
  auto replace_sch = sch;
  need_restart_ = false;

  std::set<std::string> disabled;
  for (auto &pass : passes) {
    const std::string &name = pass->GetPassName();
    const bool disable = disabled.find(name) != disabled.end();
    if (disable) {
      LOG(INFO) << "Disabling poly pass " << name;
      continue;
    } else {
      LOG(INFO) << "Running poly pass " << name;
    }

    if (LoadScheduleTreeFromFile(scop_info_.AddDumpDir(pass->GetPassName() + ".txt"), replace_sch)) {
      if (!replace_sch.plain_is_equal(final_sch)) {
        final_sch = replace_sch;
        LOG(WARNING) << (pass->GetPassName() + " input schedule had been replaced  !!!");
      }
    }

    std::stringstream time_log;
    TIMER_START;
    final_sch = pass->Run(final_sch);
    time_log << "[ Polyhedral exec time" << (scop_info_.mmu_info_.IsSpecGemm() ? "_specgemm" : "") << " ], "
             << pass->GetPassName() << " spent " << TIMER_DURATION << " ms";

    LOG(INFO) << time_log.str();
    scop_info_.RecordTime(time_log.str());

    scop_info_.DumpSchTree(pass->GetPassName(), final_sch);

    if (pass->restart_) {
      need_restart_ = true;
      break;
    }
    if (!pass->disabled_passes_.empty()) {
      disabled.insert(pass->disabled_passes_.begin(), pass->disabled_passes_.end());
      LOG(INFO) << name << " requests to disable some subsequent passes";
    }
  }
  return final_sch;
}

isl::schedule SchedulePassMgr::Run(const isl::schedule &sch, PassMgrStrategy &strategy) {
  CHECK(sch);
  strategy.RegisterPasses();
  std::vector<std::shared_ptr<SchedulePass>> passes = strategy.GetPasses();
  return Run(sch, passes);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
