/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef POLY_COMPUTE_SCHEDULE_H_
#define POLY_COMPUTE_SCHEDULE_H_

#include "poly/schedule_pass.h"

#ifdef AKG_USE_MLS
#include "poly/mls.h"
#endif

namespace akg {
namespace ir {
namespace poly {

/*
 * compute schedule pass, the main tasks ars as follow
 * 1. modify the dependences depends on the configuration switch
 * 2. Generating constraints
 * 3. According to the constraints, the ISL interface is called to generate a new schedule
 */
class ComputeSchedule : public SchedulePass {
 public:
  ComputeSchedule(PassInfo &pass_info, ScopInfo &scop_info) : pass_info_(pass_info), scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
  }
  ~ComputeSchedule() {}

  virtual isl::schedule Run(isl::schedule sch);

  void SetIslOptions();

  isl::union_map ModDependences(const isl::union_map &dependences);

  isl::schedule PermuteOuterBand(const isl::schedule sch);
  long CollectReductionExtent(const isl::union_pw_aff &aff);
  std::unordered_set<std::string> CollectSwapIds(const isl::union_pw_aff &aff, const long reduce_extent);
  isl::union_pw_aff GenerateNewAffine(const isl::union_pw_aff &swap_out, const isl::union_pw_aff &swap_in,
                                      std::unordered_set<std::string> swap_ids);
  isl::schedule AdjustInplaceAssignOrder(const isl::schedule &sch);

 private:
  PassInfo &pass_info_;

  ScopInfo &scop_info_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_COMPUTE_SCHEDULE_H_
