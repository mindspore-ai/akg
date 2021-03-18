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
#ifndef POLY_INIT_SCHEDULE_H_
#define POLY_INIT_SCHEDULE_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Init schedule pass, the main tasks ars as follow
 * 1. compute copyin
 * 2. compute dependence
 * 3. modify the dependence according to the  specific scene
 */
class InitSchedule : public SchedulePass {
 public:
  InitSchedule(PassInfo &pass_info, ScopInfo &scop_info) : pass_info_(pass_info), scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
  }
  ~InitSchedule() {}

  virtual isl::schedule Run(isl::schedule sch);

  void ComputeCopyIn(const isl::schedule &schedule);
  void RemoveUninitializedCopyin(isl::union_map &copy_in, const Binds &binds);

  void ModDependencesBeforeGroup(const isl::schedule &schedule);

  void ForceDepBetweenLiveouts(const isl::union_set liveouts);
  isl::union_map RemoveLeafSelfDependence(const isl::union_map &dependences);

 private:
  PassInfo &pass_info_;

  ScopInfo &scop_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_INIT_SCHEDULE_H_
