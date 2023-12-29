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
#ifndef POLY_COMPUTE_INNER_BAND_DEPENDENCY_H_
#define POLY_COMPUTE_INNER_BAND_DEPENDENCY_H_

#include "poly/schedule_pass.h"
#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * This class initialises the inner band dependency information used in InjectMulticoreToSchedule pass
 * and record it in the scop info. No actual schedule tree transfrom is performed in this pass.
 */
class ComputeInnerBandDependency : public SchedulePass {
 public:
  ComputeInnerBandDependency(ScopInfo &scop_info) : scop_info_(scop_info) { pass_name_ = __FUNCTION__; }
  ~ComputeInnerBandDependency() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  ScopInfo &scop_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_COMPUTE_INNER_BAND_DEPENDENCY_H_
