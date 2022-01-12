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
#include "poly/cpu_mgr_strategy.h"
#include "schedule_pass/tile_outer_band.h"
#include "schedule_pass/analyze_schedule.h"
#include "schedule_pass_cpu/cpu_memory_manager.h"
#include "schedule_pass/realize_manager.h"

namespace akg {
namespace ir {
namespace poly {

void CPUMgrStrategy::RegisterTilingPasses() { RegisterPass(std::make_shared<TileOuterBand>(pass_info_, scop_info_)); }

void CPUMgrStrategy::RegisterMemPromPasses() { RegisterPass(std::make_shared<CpuMemoryManager>(scop_info_)); }

void CPUMgrStrategy::RegisterPasses() {
  passes_.clear();
  RegisterNormalizationPasses();
  RegisterConstrainedScheduling();
  RegisterSchedulingPasses();
  RegisterPass(std::make_shared<AnalyzeSchedule>(scop_info_));
  RegisterTilingPasses();
  if (scop_info_.user_config_.GetIsTuning()) {
    return;
  }
  RegisterMemPromPasses();
  RegisterPass(std::make_shared<RealizeManager>(pass_info_, scop_info_));
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
