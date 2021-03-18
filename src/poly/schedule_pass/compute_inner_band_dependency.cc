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
#include "compute_inner_band_dependency.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule ComputeInnerBandDependency::Run(isl::schedule sch) {
  auto ori_reads = scop_info_.analysis_result_.GetReads();
  auto ori_writes = scop_info_.analysis_result_.GetWrites();
  auto ori_fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();
  auto inner_band_dependency =
    ComputeFakeCopyin(sch, ori_fake_copyin, ori_reads, ori_writes).subtract(scop_info_.analysis_result_.GetCopyin());
  scop_info_.analysis_result_.RecordInnerBandDependency(inner_band_dependency);
  return sch;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
