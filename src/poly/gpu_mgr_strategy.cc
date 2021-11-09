/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "poly/gpu_mgr_strategy.h"

#include "schedule_pass/analyze_schedule.h"
#include "schedule_pass/tile_outer_band.h"
#include "schedule_pass_gpu/mapping_outer_band.h"
#include "schedule_pass_gpu/shared_memory_manager.h"
#include "schedule_pass_gpu/register_memory_manager.h"
#include "schedule_pass_gpu/realize_manager.h"

namespace akg {
namespace ir {
namespace poly {

void GPUMgrStrategy::RegisterTilingPasses() { RegisterPass(std::make_shared<TileOuterBand>(pass_info_, scop_info_)); }

void GPUMgrStrategy::RegisterMemPromPasses() {
  RegisterPass(std::make_shared<SharedMemoryManager>(scop_info_));
  RegisterPass(std::make_shared<RegisterMemoryManager>(pass_info_, scop_info_));
}

void GPUMgrStrategy::RegisterPasses() {
  passes_.clear();
  RegisterNormalizationPasses();
  RegisterConstrainedScheduling();
  RegisterSchedulingPasses();
  RegisterPass(std::make_shared<AnalyzeSchedule>(scop_info_));
  if (scop_info_.user_config_.GetIsTuning()) {
    return;
  }
  RegisterTilingPasses();
  RegisterPass(std::make_shared<MappingOuterBand>(pass_info_, scop_info_));
  RegisterMemPromPasses();
  RegisterPass(std::make_shared<RealizeManager>(pass_info_, scop_info_));
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
